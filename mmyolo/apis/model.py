# Copyright (c) OpenMMLab. All rights reserved.
import copy
import fnmatch
import os.path as osp
import re
import warnings
from os import PathLike
from pathlib import Path
from typing import List, Tuple, Union

from mmengine.config import Config
from modelindex.load_model_index import load
from modelindex.models.Model import Model


class ModelHub:
    """A hub to host the meta information of all pre-defined models."""
    _models_dict = {}
    __mmyolo_registered = False

    @classmethod
    def register_model_index(cls,
                             model_index_path: Union[str, PathLike],
                             config_prefix: Union[str, PathLike, None] = None):
        """Parse the model-index file and register all models.

        Args:
            model_index_path (str | PathLike): The path of the model-index
                file.
            config_prefix (str | PathLike | None): The prefix of all config
                file paths in the model-index file.
        """
        model_index = load(str(model_index_path))
        model_index.build_models_with_collections()

        for metainfo in model_index.models:
            model_name = metainfo.name.lower()
            if metainfo.name in cls._models_dict:
                raise ValueError(
                    'The model name {} is conflict in {} and {}.'.format(
                        model_name, osp.abspath(metainfo.filepath),
                        osp.abspath(cls._models_dict[model_name].filepath)))
            metainfo.config = cls._expand_config_path(metainfo, config_prefix)
            cls._models_dict[model_name] = metainfo

    @classmethod
    def get(cls, model_name):
        """Get the model's metainfo by the model name.

        Args:
            model_name (str): The name of model.

        Returns:
            modelindex.models.Model: The metainfo of the specified model.
        """
        cls._register_mmyolo_models()
        # lazy load config
        metainfo = copy.deepcopy(cls._models_dict.get(model_name.lower()))
        if metainfo is None:
            raise ValueError(
                f'Failed to find model "{model_name}". please use '
                '`mmyolo.list_models` to get all available names.')
        if isinstance(metainfo.config, str):
            metainfo.config = Config.fromfile(metainfo.config)
        return metainfo

    @staticmethod
    def _expand_config_path(metainfo: Model,
                            config_prefix: Union[str, PathLike] = None):
        if config_prefix is None:
            config_prefix = osp.dirname(metainfo.filepath)

        if metainfo.config is None or osp.isabs(metainfo.config):
            config_path: str = metainfo.config
        else:
            config_path = osp.abspath(osp.join(config_prefix, metainfo.config))

        return config_path

    @classmethod
    def _register_mmyolo_models(cls):
        # register models in mmyolo
        if not cls.__mmyolo_registered:
            from importlib_metadata import distribution
            root = distribution('mmyolo').locate_file('mmyolo')
            model_index_path = root / '.mim' / 'model-index.yml'
            ModelHub.register_model_index(
                model_index_path, config_prefix=root / '.mim')
            cls.__mmyolo_registered = True

    @classmethod
    def has(cls, model_name):
        """Whether a model name is in the ModelHub."""
        return model_name in cls._models_dict


def get_model(model: Union[str, Config],
              pretrained: Union[str, bool] = False,
              device=None,
              url_mapping: Tuple[str, str] = None,
              **kwargs):
    """Get a pre-defined model or create a model from config.

    Args:
        model (str | Config): The name of model, the config file path or a
            config instance.
        pretrained (bool | str): When use name to specify model, you can
            use ``True`` to load the pre-defined pretrained weights. And you
            can also use a string to specify the path or link of weights to
            load. Defaults to False.
        device (str | torch.device | None): Transfer the model to the target
            device. Defaults to None.
        url_mapping (Tuple[str, str], optional): The mapping of pretrained
            checkpoint link. For example, load checkpoint from a local dir
            instead of download by ``('https://.*/', './checkpoint')``.
            Defaults to None.
        **kwargs: Other keyword arguments of the model config.

    Returns:
        mmengine.model.BaseModel: The result model.

    Examples:
        Get a YOLOX model and inference:

        >>> import torch
        >>> from mmyolo import get_model
        >>> inputs = torch.rand(16, 3, 640, 640)
        >>> model = get_model('yolox_s_8x8_300e_coco', pretrained=True)
        >>> feats = model(inputs)
    """  # noqa: E501
    metainfo = None
    if isinstance(model, Config):
        config = copy.deepcopy(model)
        if pretrained is True and 'load_from' in config:
            pretrained = config.load_from
    elif isinstance(model, (str, PathLike)) and Path(model).suffix == '.py':
        config = Config.fromfile(model)
        if pretrained is True and 'load_from' in config:
            pretrained = config.load_from
    elif isinstance(model, str):
        metainfo = ModelHub.get(model)
        config = metainfo.config
        if pretrained is True and metainfo.weights is not None:
            pretrained = metainfo.weights
    else:
        raise TypeError('model must be a name, a path or a Config object, '
                        f'but got {type(config)}')

    if pretrained is True:
        warnings.warn('Unable to find pre-defined checkpoint of the model.')
        pretrained = None
    elif pretrained is False:
        pretrained = None

    if kwargs:
        config.merge_from_dict({'model': kwargs})
    config.model.setdefault('data_preprocessor',
                            config.get('data_preprocessor', None))

    from mmengine.registry import DefaultScope

    from mmyolo.registry import MODELS
    with DefaultScope.overwrite_default_scope('mmyolo'):
        model = MODELS.build(config.model)

    dataset_meta = {}
    if pretrained:
        from mmengine.runner import load_checkpoint
        if url_mapping is not None:
            pretrained = re.sub(url_mapping[0], url_mapping[1], pretrained)
        checkpoint = load_checkpoint(model, pretrained, map_location='cpu')
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmpretrain 1.x
            dataset_meta = checkpoint['meta']['dataset_meta']
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # mmcls 0.x
            dataset_meta = {'classes': checkpoint['meta']['CLASSES']}

    if len(dataset_meta) == 0 and 'test_dataloader' in config:
        from mmyolo.registry import DATASETS
        dataset_class = DATASETS.get(config.test_dataloader.dataset.type)
        dataset_meta = getattr(dataset_class, 'METAINFO', {})

    model.to(device)

    model._dataset_meta = dataset_meta  # save the dataset meta
    model._config = config  # save the config in the model
    model._metainfo = metainfo  # save the metainfo in the model
    model.eval()
    return model


def list_models(pattern=None, exclude_patterns=None) -> List[str]:
    """List all models available in mmyolo.
    TODO: examples are not updated yet
    Args:
        pattern (str | None): A wildcard pattern to match model names.
            Defaults to None.
        exclude_patterns (list | None): A list of wildcard patterns to
            exclude names from the matched names. Defaults to None.
        task (str | none): The evaluation task of the model.

    Returns:
        List[str]: a list of model names.

    Examples:
        List all models:

        >>> from mmyolo import list_models
        >>> list_models()

        List YOLOv8 models on COCO dataset:

        >>> from mmyolo import list_models
        >>> list_models('yolov8*refine')
        ['yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco', 
         'yolov8_m_mask-refine_syncbn_fast_8xb16-500e_coco', 
         'yolov8_n_mask-refine_syncbn_fast_8xb16-500e_coco', 
         'yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco', 
         'yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco']

        List YOLOv5 models trained without syncbn
        YOLOv5 models:

        >>> from mmyolo import list_models
        >>> list_models('yolov5', exclude_patterns=['*syncbn'])
        ['yolov5_l-v61_fast_1xb32-50e_voc', 
         'yolov5_m-v61_fast_1xb64-50e_voc', 
         'yolov5_n-v61_fast_1xb64-50e_voc', 
         'yolov5_s-v61_fast_1xb64-50e_voc']
    """
    ModelHub._register_mmyolo_models()
    matches = set(ModelHub._models_dict.keys())

    if pattern is not None:
        # Always match keys with any postfix.
        matches = set(fnmatch.filter(matches, pattern + '*'))

    exclude_patterns = exclude_patterns or []
    for exclude_pattern in exclude_patterns:
        exclude = set(fnmatch.filter(matches, exclude_pattern + '*'))
        matches = matches - exclude

    return sorted(list(matches))
