import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import os
from scipy.stats import pearsonr


def launch_dash_scatter_plot(swact_df, power_df, info_dict, port=8050):
    external_stylesheets = ["https://fonts.googleapis.com/css2?family=Roboto&display=swap"]
    # print(os.path.join(os.path.dirname(__file__), 'assets'))
    app = dash.Dash(
        __name__,
        external_stylesheets=external_stylesheets,
        assets_folder=os.path.join(os.path.dirname(__file__), "assets"),
    )

    # Sort tables based on their design numbers
    swact_df = swact_df.sort_values(by="design_number")
    power_df = power_df.sort_values(by="design_number")

    app.layout = html.Div(
        [
            html.H2("Scatter Plot: Power vs Swact", style={"textAlign": "center"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select x column:"),
                            dcc.Dropdown(
                                id="x-column",
                                options=[{"label": col, "value": col} for col in swact_df.columns],
                                value=swact_df.swact_weighted_total.name,
                            ),
                        ],
                        style={"width": "48%", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            html.Label("Select y column:"),
                            dcc.Dropdown(
                                id="y-column",
                                options=[{"label": col, "value": col} for col in power_df.columns],
                                value=power_df.p_comb_dynamic.name,
                            ),
                        ],
                        style={"width": "48%", "display": "inline-block", "marginLeft": "4%"},
                    ),
                ]
            ),
            html.Div(id="data-info", style={"textAlign": "center"}),
            html.Div(id="title-density"),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(id="scatter-plot"),
                            html.Div(id="title-scatter", style={"textAlign": "center"}),
                        ],
                        style={"width": "48%", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="density-plot"),
                            html.Div(id="title-density", style={"textAlign": "center"}),
                        ],
                        style={"width": "48%", "display": "inline-block", "marginLeft": "4%"},
                    ),
                ]
            ),
        ]
    )

    @app.callback(
        Output("scatter-plot", "figure"),
        Output("density-plot", "figure"),
        Output("title-scatter", "children"),
        Output("title-density", "children"),
        Output("data-info", "children"),
        Input("x-column", "value"),
        Input("y-column", "value"),
    )
    def update_plot(x_col, y_col):
        x = swact_df[x_col].astype(float)
        y = power_df[y_col].astype(float)
        fig0 = px.scatter(x=x, y=y, labels={"x": x_col, "y": y_col}, template="plotly_white")
        fig0.update_traces(
            marker=dict(
                opacity=0.25,
                line=dict(
                    color="rgba(0, 0, 0, 1)",  # solid black border
                    width=0.2,
                ),
            )
        )
        title0 = html.Div(
            [
                html.Span(f"Scatter Plot of {y_col} vs {x_col}"),
            ]
        )
        fig1 = px.density_contour(x=x, y=y, labels={"x": x_col, "y": y_col})
        fig1.update_traces(contours_coloring="fill", contours_showlabels=True, colorscale="Viridis")
        title1 = html.Div(
            [
                html.Span(f"Density Plot of {y_col} vs {x_col}"),
            ]
        )

        if len(x) == len(y):
            corr, p_value = pearsonr(x.to_list(), y.to_list())
            n_points = len(x)
            info = html.Div(
                [
                    html.Div(
                        [
                            html.Span(f"Exp. Name: ", style={"fontWeight": "bold"}),
                            html.Span(f"{info_dict['experiment_name']}"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(f"Output Dir. Name: ", style={"fontWeight": "bold"}),
                            html.Span(f"{info_dict['output_dir_name']}"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(f"Technology: ", style={"fontWeight": "bold"}),
                            html.Span(f"{info_dict['technology']}"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span("Pearson correlation: ", style={"fontWeight": "bold"}),
                            html.Span(f"coeff={corr:.4f} | p={p_value:.2e}"),
                        ]
                    ),
                    html.Div([html.Span("Number of points: ", style={"fontWeight": "bold"}), html.Span(f"{n_points}")]),
                ]
            )
        else:
            info = "x and y have different lengths. Cannot compute correlation."

        return fig0, fig1, title0, title1, info

    app.run(debug=True, port=port)
