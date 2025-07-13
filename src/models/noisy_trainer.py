import torch
import torch.nn.functional as F
from transformers import Trainer


class NoisyLabelTrainer(Trainer):
    def __init__(self, noise_matrix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_matrix = torch.tensor(noise_matrix).float()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        labels_one_hot = F.one_hot(labels, num_classes=self.model.config.num_labels)
        loss = custom_loss(logits, labels_one_hot, self.noise_matrix)
        return (loss, outputs) if return_outputs else loss

def custom_loss(logits, labels, noise_matrix):
    log_probs = F.log_softmax(logits, dim=1)
    noise_adjusted_targets = torch.matmul(labels.float(), torch.tensor(noise_matrix).float().to(labels.device))
    loss = -torch.sum(noise_adjusted_targets * log_probs, dim=1)
    return loss.mean()