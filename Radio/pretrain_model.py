from torch import nn
from torchvision import models


def vgg(nb_class, weights):
    """Load pretrained model of vgg, modify classifier part."""
    print("load vgg16 bn pretrained model")
    vgg16_bn = models.vgg16_bn(weights=weights)
    vgg16_bn.classifier[-1] = nn.Linear(in_features=4096,
                                        out_features=nb_class, bias=True)
    # TODO: Init new layers, vgg16_bn.classifier[-1]

    return vgg16_bn


def resnet(nb_class, net_name, weights):
    """Load pretrained model af resnet."""

    if net_name == 'resnet18':
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(in_features=512, out_features=nb_class, bias=True)
    elif net_name == 'resnet34':
        net = models.resnet34(weights=weights)
        net.fc = nn.Linear(in_features=512, out_features=nb_class, bias=True)
    elif net_name == 'resnet50':
        net = models.resnet50(weights=weights)
        net.fc = nn.Linear(in_features=2048, out_features=nb_class, bias=True)
    elif net_name == 'resnet101':
        net = models.resnet101(weights=weights)
        net.fc = nn.Linear(in_features=2048, out_features=nb_class, bias=True)
    elif net_name == 'resnet152':
        net = models.resnet152(weights=weights)
        net.fc = nn.Linear(in_features=2048, out_features=nb_class, bias=True)
    else:
        raise ValueError(
            "Wrong pretrained model. Only resnet18/34/50/101/152 is avaliable!")

    return net


def densenet(nb_class, net_name, weights):
    """Load pretrained model of densenet."""

    if net_name == 'densenet121':
        net = models.densenet121(weights=weights)
        net.classifier = nn.Linear(in_features=1024, out_features=nb_class, bias=True)
    elif net_name == 'densenet161':
        net = models.densenet161(weights=weights)
        net.classifier = nn.Linear(in_features=2208, out_features=nb_class, bias=True)
    elif net_name == 'densenet169':
        net = models.densenet169(weights=weights)
        net.classifier = nn.Linear(in_features=1664, out_features=nb_class, bias=True)
    elif net_name == 'densenet201':
        net = models.densenet201(weights=weights)
        net.classifier = nn.Linear(in_features=1920, out_features=nb_class, bias=True)
    else:
        raise ValueError(
            "Wrong pretrained model. Only densenet121/161/169/201 is avaliable!")

    # TODO: init new layers, net.classifier
    return net


def resnext(nb_class, net_name, weights):
    """Load pretrained model of resnext."""
    if net_name == 'resnext50_32x4d':
        net = models.resnext50_32x4d(weights=weights)
    elif net_name == 'resnext101_32x8d':
        net = models.resnext101_32x8d(weights=weights)
    else:
        raise ValueError(
            "Wrong pretrained model. Only resnext50_32x4d/resnext101_32x48d is avaliable!")

    net.fc = nn.Linear(in_features=2048, out_features=nb_class, bias=True)
    # TODO: init new layers
    return net


def inception(nb_class, auxlogit=False, weights=True):
    """Load pretrained model of resnext."""
    if not auxlogit:
        net = models.inception_v3(weights=weights, aux_logits=False)
    else:
        net = models.inception_v3(weights=weights)
        net.AuxLogits.fc = nn.Linear(in_features=768, out_features=nb_class, bias=True)
    net.fc = nn.Linear(in_features=2048, out_features=nb_class, bias=True)

    # TODO: init new layers
    return net


def wide_resnet(nb_class, net_name='wide_resnet50_2', weights=True):
    """Load pretrained model of wide resnet."""
    if net_name == 'wide_resnet50_2':
        net = models.wide_resnet50_2(weights=weights)
    elif net_name == 'wide_resnet101_2':
        net = models.wide_resnet101_2(weights=weights)
    net.fc = nn.Linear(in_features=2048, out_features=nb_class, bias=True) 

    # TODO: init new layers, net.fc
    return net


def efficientnet_model(nb_class, name='efficientnet_b0', weights='DEFAULT'):
    """An easy interface of efficientnet."""
    assert name in ['efficientnet_b0',
                        'efficientnet_b1',
                        'efficientnet_b2',
                        'efficientnet_b3',
                        'efficientnet_b4',
                        'efficientnet_b5',
                        'efficientnet_b6',
                        'efficientnet_b7',
                        'efficientnet_v2_s',
                        'efficientnet_v2_m',
                        'efficientnet_v2_l'], \
                        "Efficientnet pre-define from efficientnet_b0~b7 and efficientnet_v2_s/m/l"
    net = models.get_model(name=name, weights=weights)
    
    if name in ['efficientnet_b{}'.format(str(i)) for i in [0, 1]] or name[-4:-2] == 'v2':
        net.classifier[1] = nn.Linear(1280, nb_class, bias=True)
    elif name == 'efficientnet-b2':
        net.classifier[1] = nn.Linear(1408, nb_class, bias=True)
    elif name == 'efficientnet-b3':
        net.classifier[1] = nn.Linear(1536, nb_class, bias=True)
    elif name == 'efficientnet-b4':
        net.classifier[1] = nn.Linear(1792, nb_class, bias=True)
    elif name == 'efficientnet-b5':
        net.classifier[1] = nn.Linear(2048, nb_class, bias=True)
    elif name == 'efficientnet-b6':
        net.classifier[1] = nn.Linear(2304, nb_class, bias=True)
    elif name == 'efficientnet-b7':
        net.classifier[1] = nn.Linear(2560, nb_class, bias=True)

    return net


def pretrained_model(nb_class, net_name='resnet50', weights=None):
    """Load pretrained model of torchvision.

    Args:
        nb_class: Number of class, int.
        net_name: Pretrained model's name, str. Including:
            - vgg16;
            - inceptionv3;
            - resnet18 / 34 / 50 / 101 / 152;
            - resnext50_32x4d / resnext101_32x8d;
            - wide_resnet50_2 / wide_resnet101_2.
            - densenet121 / 161 / 169 / 201;
            - efficientnet-b0~b7, v2 s / m / l
        weights: pretrain weight name or weight file directory.
    Exception:
        ValueError: Wrong pretrained model, raised when pretrained model name is
            not in net_name list.
    Return:
        model: Pretrained and initialized model.
    """
    net_name = net_name.lower()
    if net_name == 'vgg16':
        net = vgg(nb_class, weights=weights)
    elif net_name == 'inceptionv3':
        net = inception(nb_class, weights=weights)
    elif net_name[:6] == 'resnet':
        net = resnet(nb_class, net_name, weights=weights)
    elif net_name[:7] == 'resnext':
        net = resnext(nb_class, net_name, weights=weights)
    elif net_name[:11] == 'wide_resnet':
        net = wide_resnet(nb_class, net_name, weights=weights)
    elif net_name[:8] == 'densenet':
        net = densenet(nb_class, net_name, weights=weights)
    elif net_name[:12] == 'efficientnet':
        net = efficientnet_model(nb_class, net_name, weights=weights)
    else:
        raise ValueError(
            "Wrong pretrained model. Please check your pretrained model!")
    print("Load {} complete!".format(net_name))
    return net