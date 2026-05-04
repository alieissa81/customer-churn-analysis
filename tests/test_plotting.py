import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skills.utils.plotting import plot_churn_bar, save_fig, CARD_COLORS, COUNTRY_COLORS


def test_card_colors_has_all_types():
    for card in ['DIAMOND', 'GOLD', 'PLATINUM', 'SILVER']:
        assert card in CARD_COLORS, f"Missing color for {card}"


def test_country_colors_has_all_countries():
    for country in ['France', 'Germany', 'Spain']:
        assert country in COUNTRY_COLORS, f"Missing color for {country}"


def test_plot_churn_bar_returns_figure():
    data = pd.DataFrame({'Card Type': ['GOLD', 'SILVER'], 'churn_rate': [0.25, 0.15]})
    fig = plot_churn_bar(data, 'Card Type', title='Test')
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_churn_bar_with_color_map():
    data = pd.DataFrame({'Card Type': ['GOLD', 'SILVER'], 'churn_rate': [0.25, 0.15]})
    fig = plot_churn_bar(data, 'Card Type', color_map=CARD_COLORS)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_churn_bar_unknown_key_uses_fallback():
    data = pd.DataFrame({'x': ['UNKNOWN'], 'churn_rate': [0.1]})
    fig = plot_churn_bar(data, 'x', color_map=CARD_COLORS)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_save_fig_creates_file(tmp_path):
    fig, _ = plt.subplots()
    save_fig(fig, 'test_chart.png', output_dir=str(tmp_path))
    assert (tmp_path / 'test_chart.png').exists()
    plt.close(fig)
