import torch
import torch.nn.functional as F


# adversarial losses
def hinge_loss():

    def d_loss(real_pred, fake_pred):
        return (
            torch.nn.functional.relu(1 - real_pred).mean()
            + torch.nn.functional.relu(1 + fake_pred).mean()
        )

    def g_loss(fake_pred):
        return torch.nn.functional.relu(1 - fake_pred).mean()

    return d_loss, g_loss


def non_saturating_loss():

    def d_loss(real_pred, fake_pred):
        return (
            torch.nn.functional.softplus(-real_pred).mean()
            + torch.nn.functional.softplus(fake_pred).mean()
        )

    def g_loss(fake_pred):
        return torch.nn.functional.softplus(-fake_pred).mean()

    return d_loss, g_loss


def lsgan_loss():

    def d_loss(real_pred, fake_pred):
        return (
            torch.square((1 - real_pred)**2).mean()
            + torch.square(fake_pred**2).mean()
        )

    def g_loss(fake_pred):
        return torch.square((1 - fake_pred)**2).mean()

    return d_loss, g_loss


def get_label(shape, device, soft, noisy):
    real_label = torch.ones(shape, device=device)
    fake_label = torch.zeros(shape, device=device)

    if soft:
        real_label -= (torch.rand(shape, device=device) * 0.2)
        # unclear which version is better
        # real_label += (torch.rand(shape, device=device) * 0.5) - 0.3
        # fake_label += (torch.rand(shape, device=device) * 0.3)

    if noisy:
        mask1 = torch.rand(shape, device=device) > torch.tensor([0.95], device=device)
        mask2 = torch.rand(shape, device=device) > torch.tensor([0.95], device=device)
        tmp = fake_label[mask1]
        fake_label[mask2] = real_label[mask2]
        real_label[mask1] = tmp

    return real_label, fake_label



def ns_loss():
    def d_loss(real_pred, fake_pred, soft_labels=False, noisy_labels=False, reduce=None):
        real_pred, fake_pred = torch.sigmoid(real_pred), torch.sigmoid(fake_pred)
        if reduce == 'mean':
            b = real_pred.shape[0]
            real_pred, fake_pred = real_pred.view(b, -1).mean(-1), fake_pred.view(b, -1).mean(-1)
        real_label, fake_label = get_label(fake_pred.shape, fake_pred.device, soft_labels, noisy_labels)
        loss_d_real = F.binary_cross_entropy(real_pred, real_label)
        loss_d_fake = F.binary_cross_entropy(fake_pred, fake_label)
        loss_d = loss_d_fake + loss_d_real
        return loss_d

    def g_loss(fake_pred, reduce=None):
        fake_pred = torch.sigmoid(fake_pred)
        if reduce == 'mean':
            b = fake_pred.shape[0]
            fake_pred = fake_pred.view(b, -1).mean(-1)
        loss_g = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))
        return loss_g

    return d_loss, g_loss




def wgan_loss():
    def d_loss(output_real, output_fake):
        return (output_fake - output_real).mean()

    def g_loss(output_fake):
        loss = -output_fake.mean()
        return loss

    return d_loss, g_loss


def get_adversarial_losses(type="hinge"):

    ADV_LOSSES = {
        "hinge": hinge_loss,
        "non_saturating": non_saturating_loss,
        "lsgan": lsgan_loss,
        "ns": ns_loss,
        "wgan": wgan_loss
    }

    assert type.lower() in ADV_LOSSES, "Adversarial loss {type} is not implemented"
    return ADV_LOSSES[type]()


# regularizers
def r1_loss(output, input):
    grad, = torch.autograd.grad(
        outputs=output.sum(), inputs=input, create_graph=True
    )
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


def gradient_regularization(output, batch_data, center=0.0, reduction='mean'):
    batch_size = batch_data.shape[0]
    grad = torch.autograd.grad(
        # outputs=output.view(batch_size, -1).mean(-1).sum(),
        outputs=output.sum(),
        inputs=batch_data,
        create_graph=True,
        only_inputs=True
    )[0]

    grad_norm = ((grad.view(batch_size, -1).norm(2, dim=1) - center) ** 2)

    if reduction is 'max':
        grad_norm = grad_norm.max()
    elif reduction is 'sum':
        grad_norm = grad_norm.sum()
    else:
        grad_norm = grad_norm.mean()

    return grad_norm



def get_regularizer(type="r1"):

    REGULARIZERS = {
        "r1": r1_loss,
        "gradient_regularization": gradient_regularization,
    }

    assert type.lower() in REGULARIZERS, "Regularizer {type} is not implemented"
    return REGULARIZERS[type]
