def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block,layers,**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34',BasicBlock, [3,4,6,3], pretrained, progress, **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64*2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnext50_32x4d(pretrained=False, progress=True,**kwargs):
    kwargs['groups'] = 32
    kwargs['width_pre_group'] = 4

    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

