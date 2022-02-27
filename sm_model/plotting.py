import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_climate(df, settings):
    """
    For validating and displaying the indeces collected from the sensors (SWC and P).
    """

    # Figure settings
    site = df['SITE'][0]
    fig = plt.figure(figsize=(12, 6), dpi=200, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    ax.set_facecolor('#f3f3f3')
    ax2 = ax.twinx()

    # Plot settings
    if settings['smooth_plot']:
        ln1 = ax.plot(
            df.index, df['SWC'], color='dodgerblue', linestyle='dashed', label='SWC', alpha=0.5)
        ln2 = ax2.plot(
            df.index, df['TS'], color='firebrick', linestyle='dashed', label='TS', alpha=0.5)
        ln3 = ax.plot(
            df.index, df['SWC'].rolling(15).mean().shift(-3),
            color='dodgerblue', linestyle='solid', label='SWC (15-day mean)')
        ln4 = ax2.plot(
            df.index, df['TS'].rolling(15).mean().shift(-3),
            color='firebrick', linestyle='solid', label='TS (15-day mean)')

    else:
        ln1 = ax.plot(
            df.index, df['SWC'], color='dodgerblue', linestyle='solid', label='SWC')
        ln2 = ax2.plot(
            df.index, df['TS'], color='firebrick', linestyle='solid', label='TA', alpha=0.5)

    # Legend and axis settings
    legend_items = ln1 + ln2 + ln3 + ln4
    labels = [l.get_label() for l in legend_items]
    ax.legend(legend_items, labels, loc=0)

    ax.set_xlabel('')
    ax.set_ylabel('Soil water content (%)')
    ax2.set_ylabel(f'Temperature (\N{DEGREE SIGN}C)')
    ax.set_title('Climate variables for {}'.format(site), fontsize=16)

    ax.grid(True)
    ax2.grid(False)
    plt.tight_layout()
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.major.width'] = 1

    # Output settings
    source_name = settings['data_source']
    os.makedirs(os.path.join(settings['wrk_dir'], 'Plots'), exist_ok=True)
    plt.savefig(os.path.join(settings['wrk_dir'], 'Plots',
                             f'{source_name}_{site}_climate_plot.png'), dpi=200)
    plt.close(fig)
