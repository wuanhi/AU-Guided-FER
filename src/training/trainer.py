import numpy as np
import torch
from tqdm import tqdm
from src.evaluation.metrics import compute_metrics

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    scaler,
    ema,
    use_amp,
    gradient_clip,
    gradient_accumulation_steps=1,
):
    model.train()

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    batch_losses = []
    batch_accuracies = []

    all_targets = []
    all_predictions = []

    pbar = tqdm(
        dataloader,
        desc="Training",
    )

    for batch_idx, (
        inputs,
        labels,
    ) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(inputs)

            loss = loss_fn(
                logits,
                labels,
            )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        scaler.scale(loss).backward()

        if (
            batch_idx + 1
        ) % gradient_accumulation_steps == 0:

            # AMP scales gradients before backward;
            # unscale before clipping so max_norm is real.
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip,
            )

            scaler.step(optimizer)

            optimizer.zero_grad(
                set_to_none=True
            )

            scaler.update()
            ema.update()
            scheduler.step()

        batch_accuracy = (
            predictions == labels
        ).sum().item() / labels.size(0)

        batch_losses.append(
            loss.item()
        )

        batch_accuracies.append(
            batch_accuracy
        )

        all_targets.extend(
            labels.detach().cpu().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        pbar.set_postfix(
            {
                "loss": np.mean(
                    batch_losses
                ),
                "acc": np.mean(
                    batch_accuracies
                )
                * 100.0,
            }
        )

    metrics = compute_metrics(
        all_targets,
        all_predictions,
    )

    metrics["loss"] = float(
        np.mean(batch_losses)
    )

    metrics["accuracy"] = float(
        np.mean(batch_accuracies)
    )

    return metrics


@torch.no_grad()
def evaluate_one_epoch(
    model,
    dataloader,
    loss_fn,
    device,
    use_amp,
):
    model.eval()

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    batch_losses = []

    all_targets = []
    all_predictions = []

    pbar = tqdm(
        dataloader,
        desc="Validation",
    )

    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = model(inputs)

            loss = loss_fn(
                logits,
                labels,
            )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        batch_losses.append(
            loss.item()
        )

        all_targets.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

    metrics = compute_metrics(
        all_targets,
        all_predictions,
    )

    metrics["loss"] = float(
        np.mean(batch_losses)
    )

    return metrics


@torch.no_grad()
def evaluate_test_ema(
    ema,
    dataloader,
    device,
    use_amp,
):
    ema.eval()

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    all_targets = []
    all_predictions = []

    pbar = tqdm(
        dataloader,
        desc="Testing",
    )

    for inputs, labels in pbar:
        # TenCrop:
        # [B, 10, C, H, W]
        bs, ncrops, c, h, w = (
            inputs.shape
        )

        inputs = inputs.view(
            -1,
            c,
            h,
            w,
        )

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            logits = ema(inputs)

        logits = logits.view(
            bs,
            ncrops,
            -1,
        )

        outputs_avg = logits.mean(1)

        predictions = torch.argmax(
            outputs_avg,
            dim=1,
        )

        all_targets.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

    return compute_metrics(
        all_targets,
        all_predictions,
    )
