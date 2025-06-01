from .hunyuan_nodes import HunyuanPortrait_ModelLoader, HunyuanPortrait_Preprocessor, HunyuanPortrait_Generator

NODE_CLASS_MAPPINGS = {
    "HunyuanPortrait_ModelLoader": HunyuanPortrait_ModelLoader,
    "HunyuanPortrait_Preprocessor": HunyuanPortrait_Preprocessor,
    "HunyuanPortrait_Generator": HunyuanPortrait_Generator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanPortrait_ModelLoader": "Load HunyuanPortrait Models",
    "HunyuanPortrait_Preprocessor": "HunyuanPortrait Preprocessor",
    "HunyuanPortrait_Generator": "HunyuanPortrait Generator",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']