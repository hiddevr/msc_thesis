import matplotlib.pyplot as plt
from matplotlib import use as matplotlib_use
from unittest.mock import patch, MagicMock

class NonInteractivePlot:
    def __init__(self):
        self.plot_data = dict()

    def __enter__(self):
        self.original_backend = plt.get_backend()
        matplotlib_use('Agg')
        self.original_show = plt.show
        self.original_figure = plt.figure
        self.original_canvas = plt.FigureCanvasBase
        plt.show = self.dummy_show

        # Mock the figure creation and rendering process
        self.figure_patch = patch('matplotlib.pyplot.figure', self.dummy_figure)
        self.draw_patch = patch('matplotlib.backends.backend_agg.FigureCanvasAgg.draw', self.dummy_draw)
        self.tight_layout_patch = patch('matplotlib.figure.Figure.tight_layout', self.dummy_draw)

        self.figure_patch.start()
        self.draw_patch.start()
        self.tight_layout_patch.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.extract_plot_data()
        plt.show = self.original_show
        plt.figure = self.original_figure
        plt.FigureCanvasBase = self.original_canvas
        matplotlib_use(self.original_backend)

        self.figure_patch.stop()
        self.draw_patch.stop()
        self.tight_layout_patch.stop()

    def dummy_show(self, *args, **kwargs):
        pass

    def dummy_figure(self, *args, **kwargs):
        fig = self.original_figure(figsize=(1, 1))
        fig.canvas.draw = self.dummy_draw
        return fig

    def dummy_draw(self, *args, **kwargs):
        pass

    def extract_plot_data(self):
        axes = plt.gcf().get_axes()
        for ax in axes:
            for line in ax.get_lines():
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                y_label = ax.get_ylabel()

                if y_label not in self.plot_data:
                    self.plot_data[y_label] = []

                self.plot_data[y_label].append({'x': x_data, 'y': y_data})

    def get_plot_data(self):
        return self.plot_data
