"""
ANOXIA Dashboard v2.0 - Clean, Professional Layout
Optical Dead Zone Decision Support System
"""

import requests
import os
import json
import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────
NAVBAR_HEIGHT = 60
MAIN_SECTION_HEIGHT = 65  # vh
FLOW_PANEL_HEIGHT = 30    # vh
SIDEBAR_WIDTH = 30        # %
MAP_WIDTH = 70            # %

BASE_API_URL = "http://localhost:5000/api"

# Mock data for testing
MOCK_ZONES = [
    {"name": "DZ-A (Arabian Sea)", "do": 1.8, "p_hypoxia": 0.83, "status": "CRITICAL", "days": 6},
    {"name": "DZ-B (Bay of Bengal)", "do": 2.1, "p_hypoxia": 0.71, "status": "CRITICAL", "days": 8},
    {"name": "Godavari Delta", "do": 3.2, "p_hypoxia": 0.48, "status": "WARNING", "days": 14},
]

MOCK_PRECURSOR = {
    "DZ-A (Arabian Sea)": [
        ("Nitrate anomaly", "+280%", 85),
        ("Chlorophyl-a (MODIS)", "+310%", 95),
        ("Thermal stratification", "HIGH", 80),
    ],
    "DZ-B (Bay of Bengal)": [
        ("Nitrate anomaly", "+340%", 100),
        ("Chlorophyl-a (MODIS)", "+290%", 88),
        ("Thermal stratification", "MEDIUM", 55),
    ],
    "Godavari Delta": [
        ("Nitrate anomaly", "+190%", 55),
        ("Chlorophyl-a (MODIS)", "+140%", 42),
        ("Thermal stratification", "LOW", 35),
    ],
}

INTERVENTIONS = [
    {"action": "Halt fertiliser runoff", "location": "Mahanadi basin", "status": "URGENT", "color": "#ef4444"},
    {"action": "Dispatch retrieval vessel", "location": "2 ghost gears", "status": "URGENT", "color": "#f97316"},
    {"action": "Temporary no-trawl zone", "location": "reduces organic load", "status": "ACTIVE", "color": "#3b82f6"},
]

# ─────────────────────────────────────────────
# COLOR SCHEMES
# ─────────────────────────────────────────────
COLORS = {
    'dark': {
        'bg': '#0f172a',
        'card': '#1e293b',
        'text': '#f1f5f9',
        'text_muted': '#94a3b8',
        'border': '#334155',
        'red': '#ef4444',
        'orange': '#f97316',
        'green': '#22c55e',
        'blue': '#3b82f6',
        'purple': '#a855f7',
    },
    'light': {
        'bg': '#f8fafc',
        'card': '#ffffff',
        'text': '#111827',
        'text_muted': '#6b7280',
        'border': '#e5e7eb',
        'red': '#ef4444',
        'orange': '#f97316',
        'green': '#22c55e',
        'blue': '#3b82f6',
        'purple': '#a855f7',
    }
}

# ─────────────────────────────────────────────
# APP INITIALIZATION
# ─────────────────────────────────────────────
app = dash.Dash(__name__, title='ANOXIA v2 - Dead Zone Dashboard',
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap',
    ])

app.config.suppress_callback_exceptions = True

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
app.layout = html.Div(id='app-container', style={
    'width': '100vw', 'height': '100vh', 'margin': '0', 'padding': '0',
    'fontFamily': 'Inter, sans-serif', 'boxSizing': 'border-box',
    'overflow': 'hidden', 'display': 'flex', 'flexDirection': 'column',
    'background': '#0f172a', 'color': '#f1f5f9'
}, children=[
    # Data stores
    dcc.Store(id='theme-store', data='dark'),
    dcc.Store(id='flow-pathways-store', data=None),
    dcc.Interval(id='data-refresh', interval=30000, n_intervals=0),

    # ═════════════════════════════════════════════════════════════
    # NAVBAR (60px fixed)
    # ═════════════════════════════════════════════════════════════
    html.Div(id='navbar', style={
        'height': f'{NAVBAR_HEIGHT}px', 'width': '100%',
        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
        'padding': '0 24px', 'margin': '0', 'borderBottom': '1px solid #334155',
        'background': '#0f172a', 'boxSizing': 'border-box', 'flex': '0 0 auto'
    }, children=[
        html.Div([
            html.H1('ANOXIA', style={
                'margin': '0', 'fontSize': '26px', 'fontWeight': '700',
                'letterSpacing': '3px', 'color': '#f1f5f9'
            }),
            html.Div('Optical Dead Zone Decision Support', style={
                'fontSize': '11px', 'color': '#94a3b8', 'marginTop': '2px', 'letterSpacing': '0.5px'
            })
        ]),
        html.Button('🌙', id='theme-toggle-btn', n_clicks=0, style={
            'background': '#1e293b', 'border': '1px solid #334155', 'color': '#f1f5f9',
            'fontSize': '16px', 'padding': '8px 12px', 'borderRadius': '6px',
            'cursor': 'pointer', 'transition': 'all 0.3s ease'
        }),
    ]),

    # ═════════════════════════════════════════════════════════════
    # MAIN SECTION (65vh = map + sidebar)
    # ═════════════════════════════════════════════════════════════
    html.Div(id='main-section', style={
        'height': f'{MAIN_SECTION_HEIGHT}vh', 'width': '100%',
        'display': 'flex', 'margin': '0', 'padding': '0', 'gap': '0',
        'background': '#0f172a', 'boxSizing': 'border-box', 'flex': '0 0 auto'
    }, children=[
        # LEFT: Main Map (70%)
        html.Div(id='map-container', style={
            'width': f'{MAP_WIDTH}%', 'height': f'{MAIN_SECTION_HEIGHT}vh',
            'margin': '0', 'padding': '0', 'boxSizing': 'border-box',
            'position': 'relative', 'display': 'flex', 'flex': '0 0 auto'
        }, children=[
            dcc.Graph(
                id='main-map',
                style={'margin': '0', 'padding': '0', 'width': '100%', 
                       'height': '100%', 'boxSizing': 'border-box'},
                config={'displayModeBar': False, 'scrollZoom': True, 'responsive': False}
            )
        ]),

        # RIGHT: Sidebar (30%)
        html.Div(id='sidebar', style={
            'width': f'{SIDEBAR_WIDTH}%', 'height': f'{MAIN_SECTION_HEIGHT}vh',
            'margin': '0', 'padding': '12px', 'boxSizing': 'border-box',
            'overflowY': 'auto', 'overflowX': 'hidden', 'background': '#0f172a',
            'borderLeft': '1px solid #334155', 'flex': '0 0 auto'
        }, children=[
            # Zone Selector
            html.Div([
                html.Div('SELECT ZONE', style={
                    'fontSize': '10px', 'fontWeight': '700', 'letterSpacing': '1px',
                    'color': '#94a3b8', 'marginBottom': '8px', 'textTransform': 'uppercase'
                }),
                dcc.Dropdown(
                    id='zone-selector',
                    options=[{'label': z['name'], 'value': z['name']} for z in MOCK_ZONES],
                    value=MOCK_ZONES[0]['name'],
                    clearable=False,
                    style={'fontSize': '12px'}
                ),
            ], style={'background': '#1e293b', 'padding': '10px', 'marginBottom': '10px',
                     'border': '1px solid #334155', 'borderRadius': '6px', 'margin': '0'}),

            # Precursor Conditions
            html.Div([
                html.Div('PRECURSOR CONDITIONS', style={
                    'fontSize': '10px', 'fontWeight': '700', 'letterSpacing': '1px',
                    'color': '#94a3b8', 'marginBottom': '8px', 'textTransform': 'uppercase'
                }),
                html.Div(id='precursor-list', children=[])
            ], style={'background': '#1e293b', 'padding': '10px', 'marginBottom': '10px',
                     'border': '1px solid #334155', 'borderRadius': '6px', 'margin': '0'}),

            # Interventions
            html.Div([
                html.Div('RECOMMENDED ACTIONS', style={
                    'fontSize': '10px', 'fontWeight': '700', 'letterSpacing': '1px',
                    'color': '#94a3b8', 'marginBottom': '8px', 'textTransform': 'uppercase'
                }),
                html.Div(id='interventions-list', children=[
                    html.Div([
                        html.Div(i['action'], style={
                            'fontSize': '12px', 'fontWeight': '600', 'color': '#f1f5f9',
                            'marginBottom': '4px'
                        }),
                        html.Div(f"📍 {i['location']}", style={
                            'fontSize': '11px', 'color': '#94a3b8', 'marginBottom': '4px'
                        }),
                        html.Div(f"Status: {i['status']}", style={
                            'fontSize': '10px', 'color': i['color'], 'fontWeight': '600'
                        }),
                    ], style={
                        'padding': '8px', 'marginBottom': '8px', 'borderLeft': f'3px solid {i["color"]}',
                        'background': '#0f172a', 'borderRadius': '3px'
                    })
                    for i in INTERVENTIONS
                ])
            ], style={'background': '#1e293b', 'padding': '10px',
                     'border': '1px solid #334155', 'borderRadius': '6px', 'margin': '0'}),

            # Active Zones
            html.Div([
                html.Div('ACTIVE THREAT ZONES', style={
                    'fontSize': '10px', 'fontWeight': '700', 'letterSpacing': '1px',
                    'color': '#94a3b8', 'marginBottom': '8px', 'textTransform': 'uppercase'
                }),
                html.Div(id='active-zones-list', children=[
                    html.Div([
                        html.Div([
                            html.Span(z['name'], style={'color': '#f1f5f9', 'fontWeight': '600'}),
                            html.Span(z['status'], style={
                                'background': '#ef4444' if z['status'] == 'CRITICAL' else '#f97316',
                                'color': '#fff', 'fontSize': '9px', 'padding': '2px 6px',
                                'borderRadius': '3px', 'marginLeft': '8px', 'fontWeight': '600'
                            }),
                        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '4px'}),
                        html.Div([
                            html.Span(f"DO: {z['do']} mg/L", style={'fontSize': '11px', 'color': '#94a3b8', 'marginRight': '8px'}),
                            html.Span(f"P(hypoxia): {int(z['p_hypoxia']*100)}%", style={
                                'fontSize': '11px', 'fontWeight': '600',
                                'color': '#ef4444' if z['status'] == 'CRITICAL' else '#f97316'
                            }),
                        ], style={'display': 'flex', 'fontSize': '10px'}),
                    ], style={
                        'padding': '8px', 'marginBottom': '8px', 'background': '#0f172a',
                        'border': f'1px solid {"#ef4444" if z["status"] == "CRITICAL" else "#f97316"}',
                        'borderRadius': '4px'
                    })
                    for z in MOCK_ZONES
                ])
            ], style={'background': '#1e293b', 'padding': '10px',
                     'border': '1px solid #334155', 'borderRadius': '6px', 'margin': '0'}),
        ]),
    ]),

    # ═════════════════════════════════════════════════════════════
    # FLOW PANEL (30vh = second map)
    # ═════════════════════════════════════════════════════════════
    html.Div(id='flow-panel', style={
        'height': f'{FLOW_PANEL_HEIGHT}vh', 'width': '100%',
        'margin': '0', 'padding': '12px', 'boxSizing': 'border-box',
        'background': '#0f172a', 'borderTop': '1px solid #334155',
        'display': 'flex', 'flexDirection': 'column', 'flex': '0 0 auto'
    }, children=[
        # Header
        html.Div([
            html.Span('🌊 Runoff → Hypoxia Flow Pathways', style={
                'fontSize': '13px', 'fontWeight': '700', 'color': '#f1f5f9',
                'letterSpacing': '0.5px'
            }),
            html.Span('Nutrient transport from agricultural runoff to dead zones', style={
                'fontSize': '10px', 'color': '#94a3b8', 'marginLeft': '8px'
            })
        ], style={'margin': '0', 'padding': '0 0 8px 0', 'display': 'flex', 'alignItems': 'center'}),

        # Flow Map
        html.Div([
            dcc.Graph(
                id='flow-pathways-map',
                style={'margin': '0', 'padding': '0', 'width': '100%', 
                       'height': '100%', 'boxSizing': 'border-box'},
                config={'displayModeBar': False, 'scrollZoom': True, 'responsive': False}
            )
        ], style={
            'flex': '1', 'width': '100%', 'margin': '0', 'padding': '0',
            'boxSizing': 'border-box', 'border': '1px solid #334155', 'borderRadius': '6px'
        }),

        # Flow Legend
        html.Div(id='flow-legend', style={
            'marginTop': '8px', 'display': 'flex', 'gap': '12px', 'fontSize': '10px',
            'margin': '0', 'padding': '0'
        }, children=[
            html.Div([
                html.Div(style={'width': '12px', 'height': '12px', 'background': '#fbbf24', 'borderRadius': '2px'}),
                'Weak (0-33%)'
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px', 'color': '#94a3b8'}),
            html.Div([
                html.Div(style={'width': '12px', 'height': '12px', 'background': '#f97316', 'borderRadius': '2px'}),
                'Medium (33-66%)'
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px', 'color': '#94a3b8'}),
            html.Div([
                html.Div(style={'width': '12px', 'height': '12px', 'background': '#ef4444', 'borderRadius': '2px'}),
                'Strong (66-100%)'
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px', 'color': '#94a3b8'}),
        ]),
    ]),
])

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────

@callback(
    [Output('precursor-list', 'children'),
     Output('active-zones-list', 'children')],
    Input('zone-selector', 'value')
)
def update_sidebar(selected_zone):
    """Update sidebar with selected zone data"""
    
    # Precursor conditions
    precursor_data = MOCK_PRECURSOR.get(selected_zone, [])
    precursor_children = [
        html.Div([
            html.Div(label, style={'fontSize': '11px', 'fontWeight': '500', 'color': '#f1f5f9', 'marginBottom': '3px'}),
            html.Div(style={
                'width': '100%', 'height': '4px', 'background': '#334155', 'borderRadius': '2px',
                'overflow': 'hidden', 'marginBottom': '6px'
            }, children=[
                html.Div(style={
                    'width': f'{pct}%', 'height': '100%',
                    'background': '#ef4444' if pct >= 80 else '#f97316' if pct >= 50 else '#22c55e',
                    'transition': 'width 0.3s ease'
                })
            ]),
            html.Div(value_text, style={'fontSize': '10px', 'color': '#94a3b8'})
        ], style={'marginBottom': '8px'})
        for label, value_text, pct in precursor_data
    ]
    
    # Active zones
    active_zones = MOCK_ZONES
    active_children = [
        html.Div([
            html.Div([
                html.Span(z['name'], style={'color': '#f1f5f9', 'fontWeight': '600', 'fontSize': '11px'}),
                html.Span(z['status'], style={
                    'background': '#ef4444' if z['status'] == 'CRITICAL' else '#f97316',
                    'color': '#fff', 'fontSize': '8px', 'padding': '2px 5px',
                    'borderRadius': '2px', 'fontWeight': '600'
                }),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '4px'}),
            html.Div([
                html.Span(f"DO: {z['do']}", style={'fontSize': '10px', 'color': '#94a3b8', 'marginRight': '6px'}),
                html.Span(f"{int(z['p_hypoxia']*100)}%", style={
                    'fontSize': '10px', 'fontWeight': '600',
                    'color': '#ef4444' if z['status'] == 'CRITICAL' else '#f97316'
                }),
            ], style={'display': 'flex'}),
        ], style={
            'padding': '7px', 'marginBottom': '7px', 'background': '#0f172a',
            'border': f'1px solid {"#ef4444" if z["status"] == "CRITICAL" else "#f97316"}',
            'borderRadius': '3px', 'fontSize': '10px'
        })
        for z in active_zones
    ]
    
    return precursor_children, active_children


@callback(
    Output('main-map', 'figure'),
    Input('theme-store', 'data')
)
def update_main_map(theme):
    """Create main hypoxia map"""
    colors = COLORS.get(theme, COLORS['dark'])
    
    fig = go.Figure()
    
    # Dead zone markers
    fig.add_trace(go.Scattermapbox(
        lon=[66, 82, 63.5],
        lat=[15, 12.5, 13],
        mode='markers',
        marker=dict(size=15, color=['#ef4444', '#ef4444', '#f97316'], opacity=0.8),
        text=['DZ-A<br>P=83%', 'DZ-B<br>P=71%', 'Delta<br>P=48%'],
        hovertemplate='%{text}<extra></extra>',
        name='Dead Zones'
    ))
    
    # Ghost gear markers
    fig.add_trace(go.Scattermapbox(
        lon=[65.5, 84],
        lat=[14.5, 11.5],
        mode='markers+text',
        marker=dict(size=10, color='#a855f7', symbol='diamond', opacity=0.6),
        text=['👻', '👻'],
        textposition='top center',
        hovertemplate='Ghost gear convergence<extra></extra>',
        name='Ghost Gear'
    ))
    
    # Hypoxia heatmap
    lons = np.linspace(62, 86, 20)
    lats = np.linspace(9, 17, 20)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Synthetic hypoxia probability
    z_values = np.exp(-((lon_grid - 66)**2 + (lat_grid - 15)**2) / 5) * 0.8
    z_values += np.exp(-((lon_grid - 82)**2 + (lat_grid - 12.5)**2) / 5) * 0.7
    
    fig.add_trace(go.Densitymapbox(
        lon=lon_grid.flatten(),
        lat=lat_grid.flatten(),
        z=z_values.flatten(),
        colorscale='Reds',
        opacity=0.4,
        hovertemplate='Hypoxia Probability: %{z:.2%}<extra></extra>',
        name='Hypoxia Risk',
        colorbar=dict(title='P(Hypoxia)', thickness=10, len=0.7)
    ))
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=13.2, lon=74),
            zoom=5.5
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='closest',
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(family='Inter, sans-serif', color=colors['text'], size=11)
    )
    
    return fig


@callback(
    Output('flow-pathways-map', 'figure'),
    [Input('flow-pathways-store', 'data'),
     Input('theme-store', 'data')]
)
def update_flow_map(flow_data, theme):
    """Create runoff to hypoxia flow pathways map"""
    colors = COLORS.get(theme, COLORS['dark'])
    
    fig = go.Figure()
    
    # Define rivers
    rivers = [
        {
            'name': 'Godavari',
            'source': [16.5, 81.5],
            'path': [[16.5, 81.5], [16.0, 80], [15.5, 78.5], [15.0, 76.5], [14.5, 74.5], [14.0, 72], [13.5, 70]],
            'end': [13.5, 70],
            'strength': 0.75,
        },
        {
            'name': 'Krishna',
            'source': [16.2, 80.8],
            'path': [[16.2, 80.8], [15.8, 79.2], [15.2, 77.5], [14.8, 75.2], [14.2, 72.8], [13.8, 70.5]],
            'end': [13.8, 70.5],
            'strength': 0.58,
        },
        {
            'name': 'Mahanadi',
            'source': [20.5, 86],
            'path': [[20.5, 86], [18.5, 83], [16.5, 81], [15.0, 79], [14.0, 77], [13.0, 74]],
            'end': [13.0, 74],
            'strength': 0.52,
        },
    ]
    
    # Draw pathways
    for river in rivers:
        path = river['path']
        strength = river['strength']
        
        # Color by strength
        if strength >= 0.66:
            color = '#ef4444'  # red
        elif strength >= 0.33:
            color = '#f97316'  # orange
        else:
            color = '#fbbf24'  # yellow
        
        # Draw line
        lons = [p[1] for p in path]
        lats = [p[0] for p in path]
        
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode='lines',
            line=dict(width=4, color=color),
            hovertemplate=f'{river["name"]}<br>Strength: {strength:.0%}<extra></extra>',
            name=f'{river["name"]} ({strength:.0%})',
            opacity=0.8
        ))
        
        # Start marker (green)
        fig.add_trace(go.Scattermapbox(
            lon=[river['source'][1]],
            lat=[river['source'][0]],
            mode='markers',
            marker=dict(size=8, color='#22c55e'),
            hovertemplate='River source<extra></extra>',
            showlegend=False,
            name=''
        ))
        
        # End marker (red)
        fig.add_trace(go.Scattermapbox(
            lon=[river['end'][1]],
            lat=[river['end'][0]],
            mode='markers',
            marker=dict(size=8, color='#ef4444'),
            hovertemplate='Convergence zone<extra></extra>',
            showlegend=False,
            name=''
        ))
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=15.5, lon=76),
            zoom=5
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='closest',
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        font=dict(family='Inter, sans-serif', color=colors['text'], size=10),
        showlegend=False
    )
    
    return fig


@callback(
    Output('theme-store', 'data'),
    Input('theme-toggle-btn', 'n_clicks'),
    State('theme-store', 'data')
)
def toggle_theme(n_clicks, current_theme):
    """Toggle between dark and light theme"""
    if n_clicks % 2 == 0:
        return 'dark'
    return 'light'


@callback(
    Output('navbar', 'style'),
    Input('theme-store', 'data')
)
def update_navbar_theme(theme):
    """Update navbar colors based on theme"""
    colors = COLORS.get(theme, COLORS['dark'])
    return {
        'height': f'{NAVBAR_HEIGHT}px', 'width': '100%',
        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
        'padding': '0 24px', 'margin': '0', 'borderBottom': f'1px solid {colors["border"]}',
        'background': colors['bg'], 'boxSizing': 'border-box', 'flex': '0 0 auto'
    }


if __name__ == '__main__':
    logger.info("🚀 Starting ANOXIA v2.0 Dashboard on port 8000...")
    app.run(debug=True, host='0.0.0.0', port=8000, dev_tools_ui=True)
