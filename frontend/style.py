"""
style.py
========

This module defines custom Gradio theme classes for the Latxa-Instruct frontend.
It provides a `White` theme that customizes the appearance of Gradio components,
including colors, fonts, spacing, and button styles. The theme is designed to
offer a clean and modern user interface, with options for further customization.
It also includes logic to ensure compatibility with different Gradio versions.

License:
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at:

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Author:
    Oscar Sainz (oscar.sainz@ehu.eus)
"""
from packaging.version import Version

import gradio
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

class White(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color  = colors.emerald,
        secondary_hue: colors.Color  = colors.blue,
        neutral_hue: colors.Color  = colors.gray,
        spacing_size: sizes.Size  = sizes.spacing_md,
        radius_size: sizes.Size  = sizes.radius_md,
        text_size: sizes.Size  = sizes.text_md,
        font: fonts.Font = (
            fonts.GoogleFont("Poppins"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            # body_background_fill="repeating-linear-gradient(45deg, *primary_200, *primary_200 10px, *primary_50 10px, *primary_50 20px)",
            # body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 10px, *primary_900 10px, *primary_900 20px)",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            # button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="1px",
            block_shadow="*shadow_drop_sm",
            # button_primary_shadow="*shadow_drop_lg",
            # button_large_padding="32px",
        )


    def _get_theme_css(self):
        """Remove dark theme styles from the theme"""
        css = super()._get_theme_css()
        if Version(gradio.get_package_version()) < Version("5.0.0"):
            light_css = css.split('.dark')[0]
            css = "\n".join([light_css, light_css.replace(':root', '.dark')])
        else:
            css_splits = css.split(':root')
            css_splits[2] = css_splits[1].rstrip("@media (prefers-color-scheme: dark) {\n")
            css = ":root".join(css_splits)
        return css

    
if __name__ == "__main__":
    White()
