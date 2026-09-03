import torch

import coca_pytorch.coca_pytorch as coca_module


def test_contrastive_loss_uses_gathered_batch_size(monkeypatch):
    """Distributed contrastive targets must match the gathered latent rows."""
    torch.manual_seed(0)
    model = coca_module.CoCa(
        dim=8,
        num_tokens=16,
        unimodal_depth=0,
        multimodal_depth=0,
        image_dim=8,
        num_img_queries=1,
        dim_head=4,
        heads=2,
    )

    # Avoid starting a process group while still exercising the distributed
    # branch.  The extra row represents a sample gathered from another rank;
    # detaching it mirrors AllGather.backward, which returns only local rows.
    def fake_all_gather(latents):
        remote_latent = latents[:1].detach().clone()
        return torch.cat((latents, remote_latent), dim=0)

    model.is_distributed = True
    monkeypatch.setattr(coca_module, "all_gather", fake_all_gather)

    text = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    image_tokens = torch.randn(2, 3, 8)

    loss = model(text=text, image_tokens=image_tokens, return_loss=True)
    assert torch.isfinite(loss)

    loss.backward()
    assert model.temperature.grad is not None
    assert torch.isfinite(model.temperature.grad).all()
