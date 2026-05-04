import os
import matplotlib.pyplot as plt


CARD_COLORS = {
    'DIAMOND': '#1565C0',
    'GOLD': '#F9A825',
    'PLATINUM': '#757575',
    'SILVER': '#90A4AE',
}

COUNTRY_COLORS = {
    'France': '#1565C0',
    'Germany': '#B71C1C',
    'Spain': '#E65100',
}


def save_fig(fig, filename, output_dir):
    """Save figure to output_dir/filename, creating the directory if needed."""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=150)


def plot_churn_bar(data, x_col, churn_col='churn_rate', title='', color_map=None):
    """Bar chart of churn rate by a categorical column."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = (
        [color_map.get(v, '#607D8B') for v in data[x_col]]
        if color_map else '#607D8B'
    )
    ax.bar(data[x_col], data[churn_col] * 100, color=colors, edgecolor='white')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title(title)
    ax.set_ylim(0, min(max(data[churn_col].max() * 130, 5), 100))
    return fig
