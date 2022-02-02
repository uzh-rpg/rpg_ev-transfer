import torch
import torch.nn.functional as F


def squared_error(input, target):
    return torch.sum((input - target) ** 2)


def generator_loss(loss_func, fake):
    if loss_func == "wgan":
        return -fake.mean()
    elif loss_func == "gan":
        return F.binary_cross_entropy_with_logits(input=fake, target=torch.ones_like(fake).cuda())
    elif loss_func == "lsgan":
        return squared_error(input=fake, target=torch.ones_like(fake).cuda()).mean() 
    elif loss_func == "hinge":
        return -fake.mean()
    else:
        raise Exception("Invalid loss_function")


def generator_loss_two_sensors(loss_func, sensor_a=None, sensor_b=None):
    if loss_func == "hinge":
        if sensor_a is None:
            return -sensor_b.mean()
        elif sensor_b is None:
            return sensor_a.mean()
        else:
            return sensor_a.mean() - sensor_b.mean()
    else:
        raise Exception("Invalid loss_function")


def discriminator_loss(loss_func, real=torch.ones([1]), fake=-torch.ones([1])):
    if loss_func == "wgan":
        real_loss = -real.mean()
        fake_loss = fake.mean()
    elif loss_func == "gan":
        real_loss = F.binary_cross_entropy_with_logits(input=real,
                                                       target=torch.ones_like(real).cuda())
        fake_loss = F.binary_cross_entropy_with_logits(input=fake,
                                                       target=torch.zeros_like(fake).cuda())
    elif loss_func == "lsgan":
        real_loss = squared_error(input=real, target=torch.ones_like(real).cuda()).mean()
        fake_loss = squared_error(input=fake, target=torch.zeros_like(fake).cuda()).mean()
    elif loss_func == "hinge":
        real_loss = F.relu(1.0 - real).mean()
        fake_loss = F.relu(1.0 + fake).mean()
    else:
        raise Exception("Invalid loss_function")

    return real_loss + fake_loss


def event_reconstruction_loss(gt_histogram, predicted_histogram):
    l1_distance = torch.abs(gt_histogram - predicted_histogram).sum(dim=1)
    bool_zero_cells = gt_histogram.sum(dim=1) > 0

    if torch.logical_not(bool_zero_cells).sum() == 0 or bool_zero_cells.sum() == 0:
        return l1_distance.mean()

    return l1_distance[bool_zero_cells].mean() + l1_distance[torch.logical_not(bool_zero_cells)].mean()


def normalize_l2(x):
    return torch.nn.functional.normalize(x.flatten(start_dim=-2), p=2, dim=1).reshape_as(x)
