import requests
import os
import json
import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
REAL_DATA_PATH = '../outputs/dashboard_data.json'
MOCK_DATA = {
    "dz_lats": [14.5, 15.0, 15.5, 16.0, 14.0, 13.5, 13.0, 12.5,
                12.0, 11.5, 11.0, 10.5, 10.0, 9.5, 9.0, 8.5],
    "dz_lons": [65.0, 65.5, 66.0, 66.5, 64.5, 64.0, 63.5, 63.0,
                82.0, 82.5, 83.0, 83.5, 84.0, 84.5, 85.0, 85.5],
    "dz_probs": [0.83, 0.81, 0.79, 0.76, 0.80, 0.77, 0.74, 0.71,
                 0.72, 0.75, 0.78, 0.80, 0.82, 0.79, 0.76, 0.73],
    "drift_paths": [
        {"id": 0, "lats": [16.0, 15.8, 15.5, 15.1, 14.7, 14.4, 14.1, 13.8],
         "lons": [72.0, 71.5, 71.0, 70.3, 69.5, 68.7, 67.8, 66.9]},
        {"id": 1, "lats": [17.2, 17.0, 16.7, 16.3, 15.9, 15.5, 15.1, 14.7],
         "lons": [73.5, 73.0, 72.4, 71.7, 71.0, 70.2, 69.3, 68.4]},
        {"id": 2, "lats": [14.0, 13.8, 13.5, 13.2, 12.9, 12.6, 12.3, 12.0],
         "lons": [80.5, 81.0, 81.5, 82.0, 82.5, 83.0, 83.4, 83.8]},
        {"id": 3, "lats": [15.5, 15.2, 14.9, 14.5, 14.1, 13.7, 13.3, 12.9],
         "lons": [79.0, 79.5, 80.0, 80.6, 81.2, 81.8, 82.3, 82.7]},
    ],
    "traps": [
        {"lat": 14.1, "lon": 66.9, "p_hypoxia": 0.83, "severity": 2.82, "window_days": 8},
        {"lat": 12.6, "lon": 83.0, "p_hypoxia": 0.78, "severity": 2.65, "window_days": 8},
    ],
    "zones": [
        {"name": "DZ-A (Arabian Sea)",   "do": 1.8, "p_hypoxia": 0.83,
         "gear_paths": 2, "days": 6,  "status": "CRITICAL"},
        {"name": "DZ-B (Bay of Bengal)", "do": 2.1, "p_hypoxia": 0.71,
         "gear_paths": 2, "days": 8,  "status": "CRITICAL"},
        {"name": "Godavari Delta",       "do": 3.2, "p_hypoxia": 0.48,
         "gear_paths": 0, "days": 14, "status": "WARNING"},
    ]
}

PRECURSOR_DATA = {
    "DZ-A (Arabian Sea)": [
        ("Nitrate anomaly",        "+280%", 85),
        ("Chlorophyl-a (MODIS)",   "+310%", 95),
        ("Thermal stratification", "HIGH",  80),
        ("Wind stress mixing",     "LOW",   25),
        ("DO drawdown rate",       "FAST",  90),
    ],
    "DZ-B (Bay of Bengal)": [
        ("Nitrate anomaly",        "+340%", 100),
        ("Chlorophyl-a (MODIS)",   "+290%", 88),
        ("Thermal stratification", "MEDIUM",55),
        ("Wind stress mixing",     "LOW",   30),
        ("DO drawdown rate",       "FAST",  85),
    ],
    "Godavari Delta": [
        ("Nitrate anomaly",        "+190%", 55),
        ("Chlorophyl-a (MODIS)",   "+140%", 42),
        ("Thermal stratification", "LOW",   35),
        ("Wind stress mixing",     "MEDIUM",50),
        ("DO drawdown rate",       "SLOW",  25),
    ],
}

INTERVENTIONS = [
    {"color": "critical", "action": "Halt fertiliser runoff", "location": "Mahanadi basin",
     "agency": "Agriculture Ministry", "status": "URGENT", "timeline": "5 days"},
    {"color": "warning", "action": "Dispatch retrieval vessel", "location": "2 ghost gears converging",
     "agency": "Coast Guard", "status": "URGENT", "timeline": "8-day window"},
    {"color": "info", "action": "Temporary no-trawl zone", "location": "reduces organic load",
     "agency": "Odisha Fisheries", "status": "URGENT", "timeline": "14 days"},
    {"color": "ghost", "action": "Ghost gear paths", "location": "monitoring — 4 active",
     "agency": "Agriculture Ministry", "status": "ACTIVE", "timeline": "5 days"},
    {"color": "success", "action": "Auto-file SDG 14.1 incident report", "location": "",
     "agency": "MoEFCC", "status": "GENERATED", "timeline": "2 days"},
]

def load_data():
    if os.path.exists(REAL_DATA_PATH):
        try:
            with open(REAL_DATA_PATH) as f:
                return json.load(f)
        except:
            return MOCK_DATA
    return MOCK_DATA

DATA = load_data()

# ─────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────
app = dash.Dash(__name__, title='ANOXIA Dashboard', 
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap',
    ])

app.config.suppress_callback_exceptions = True

# ─────────────────────────────────────────────
# THEME COLORS
# ─────────────────────────────────────────────
def get_color_scheme(theme):
    if theme == 'light':
        return {
            'bg': '#f8fafc',
            'card': '#ffffff',
            'text': '#111827',
            'text_muted': '#6b7280',
            'border': '#e5e7eb',
            'red': '#ef4444',
            'orange': '#f97316',
            'purple': '#a855f7',
            'green': '#22c55e',
            'blue': '#3b82f6',
            'map_style': 'open-street-map',
        }
    else:  # dark
        return {
            'bg': '#0f172a',
            'card': '#1e293b',
            'text': '#f1f5f9',
            'text_muted': '#94a3b8',
            'border': '#334155',
            'red': '#ef4444',
            'orange': '#f97316',
            'purple': '#a855f7',
            'green': '#22c55e',
            'blue': '#3b82f6',
            'map_style': 'open-street-map',
        }

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgba_str(hex_color, alpha=0.9):
    """Convert hex to rgba string"""
    r, g, b = hex_to_rgb(hex_color)
    return f'rgba({r},{g},{b},{alpha})'

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def kpi_card(label, value, color, theme):
    colors = get_color_scheme(theme)
    return html.Div([
        html.Div(value, style={
            'fontSize': '32px', 'fontWeight': '800', 'color': colors[color],
            'marginBottom': '6px', 'letterSpacing': '-1px'
        }),
        html.Div(label, style={
            'fontSize': '11px', 'color': colors['text_muted'],
            'textTransform': 'uppercase', 'letterSpacing': '1.2px', 'fontWeight': '600'
        }),
    ], className='kpi-card', style={
        'background': colors['card'],
        'borderRadius': '8px', 'padding': '16px', 'textAlign': 'center',
        'flex': '1', 'border': f'1px solid {colors["border"]}',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'transition': 'all 0.3s ease'
    })

def precursor_bar(label, value_text, pct, theme):
    colors = get_color_scheme(theme)
    if pct >= 80:
        bar_color = colors['red']
    elif pct >= 50:
        bar_color = colors['orange']
    else:
        bar_color = colors['green']
    
    return html.Div([
        html.Div([
            html.Span(label, style={'color': colors['text'], 'fontSize': '13px'}),
            html.Span(value_text, style={'color': bar_color, 'fontSize': '13px',
                'fontWeight': '600', 'float': 'right'}),
        ], style={'marginBottom': '6px', 'overflow': 'hidden', 'display': 'flex',
            'justifyContent': 'space-between'}),
        html.Div(
            html.Div(style={'width': f'{pct}%', 'height': '6px',
                'background': bar_color, 'borderRadius': '3px',
                'transition': 'width 0.5s ease', 'boxShadow': f'0 0 8px {bar_color}80'}),
            style={'background': colors['border'], 'borderRadius': '3px', 'height': '6px',
                'marginBottom': '14px', 'overflow': 'hidden'}
        )
    ])

def zone_card(zone, theme):
    colors = get_color_scheme(theme)
    status_color = colors['red'] if zone['status'] == 'CRITICAL' else colors['orange']
    
    return html.Div([
        html.Div([
            html.Span(zone['name'], style={'color': colors['text'], 'fontWeight': '700',
                'fontSize': '14px'}),
            html.Span(zone['status'], style={'background': status_color, 'color': colors['card'],
                'fontSize': '10px', 'padding': '4px 10px', 'borderRadius': '6px',
                'marginLeft': '10px', 'fontWeight': '700'}),
        ], style={'marginBottom': '10px', 'display': 'flex', 'justifyContent': 'space-between'}),
        html.Div([
            html.Span(f"DO: {zone['do']} mg/L", style={'color': colors['text_muted'],
                'fontSize': '12px', 'marginRight': '12px'}),
            html.Span(f"P(hypoxia): {int(zone['p_hypoxia']*100)}%", style={'color': status_color,
                'fontSize': '12px', 'marginRight': '12px', 'fontWeight': '600'}),
            html.Span(f"Paths: {zone['gear_paths']}", style={'color': colors['purple'],
                'fontSize': '12px', 'marginRight': '12px'}),
            html.Span(f"{zone['days']}d", style={'color': colors['orange'], 'fontSize': '12px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap'})
    ], className='zone-card', style={
        'background': colors['card'], 'borderLeft': f'4px solid {status_color}',
        'borderRadius': '8px', 'padding': '12px', 'marginBottom': '8px',
        'border': f'1px solid {colors["border"]}', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'cursor': 'pointer', 'transition': 'all 0.3s ease'
    })

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
app.layout = html.Div(id='main-container', style={
    'background': '#0f172a', 'height': '100vh', 'fontFamily': 'Inter, system-ui, sans-serif',
    'color': '#f1f5f9', 'transition': 'background 0.3s ease, color 0.3s ease',
    'display': 'flex', 'flexDirection': 'column', 'margin': '0', 'padding': '0',
    'overflow': 'hidden', 'boxSizing': 'border-box'
}, children=[
    dcc.Store(id='theme-store', data='dark'),
    dcc.Store(id='data-store', data=DATA),
    dcc.Store(id='map-click-store', data=None),  # Store for map click coordinates
    dcc.Store(id='wind-vectors-store', data=None),  # Wind vectors cache
    dcc.Store(id='ocean-currents-store', data=None),  # Ocean currents cache
    dcc.Interval(id='data-refresh', interval=300000, n_intervals=0),  # Refresh every 5 minutes instead of 10 seconds

    # NAVBAR
    html.Div(id='navbar', style={
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
        'padding': '12px 24px', 'borderBottom': '1px solid #334155', 'background': '#0f172a',
        'transition': 'all 0.3s ease', 'margin': '0', 'height': '60px', 'boxSizing': 'border-box', 'flex': '0 0 auto'
    }, children=[
        html.Div([
            html.H1('ANOXIA', id='anoxia-title', style={'margin': '0', 'fontSize': '28px',
                'letterSpacing': '4px', 'fontWeight': '800'}),
            html.Div('Bay of Bengal · Dead zones + ghost gear overlay', id='navbar-subtitle',
                style={'fontSize': '12px', 'marginTop': '2px'}),
        ]),
        
        html.Div([
            dcc.RadioItems(
                id='layer-toggle', value='all',
                options=[{'label': ' ⚠️ All Layers','value': 'all'},
                         {'label': ' 🌊 Dead Zones Only','value': 'dz'},
                         {'label': ' 👻 Ghost Gear Only','value': 'gg'}],
                inline=True, inputStyle={'marginRight': '6px', 'marginLeft': '18px'},
                style={'fontSize': '12px', 'display': 'flex',
                    'gap': '2px', 'marginRight': '24px', 'transition': 'color 0.3s ease'}
            ),
            
            # Overlay Toggle Buttons
            html.Div([
                html.Button('💨 Wind Patterns', id='toggle-wind-btn', n_clicks=0, title='Show wind vectors', style={
                    'fontSize': '12px', 'fontWeight': '600', 'padding': '8px 14px', 'borderRadius': '8px',
                    'cursor': 'pointer', 'transition': 'all 0.3s ease', 'marginRight': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
                }),
                html.Button('🌀 Ocean Currents', id='toggle-currents-btn', n_clicks=0, title='Show ocean currents', style={
                    'fontSize': '12px', 'fontWeight': '600', 'padding': '8px 14px', 'borderRadius': '8px',
                    'cursor': 'pointer', 'transition': 'all 0.3s ease', 'marginRight': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
                }),
            ], style={'display': 'flex', 'gap': '12px', 'marginRight': '16px', 'flexWrap': 'wrap'}),
            
            # Dark/Light Toggle
            html.Button('🌙', id='theme-toggle', style={
                'fontSize': '18px', 'padding': '8px 14px', 'borderRadius': '8px',
                'cursor': 'pointer', 'transition': 'all 0.3s ease',
                'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
            }),
        ], style={'display': 'flex', 'alignItems': 'center'}),
    ]),



    # MAIN CONTENT WRAPPER (Map + Sidebar + Flow Panel)
    html.Div(id='content-wrapper', style={
        'background': '#0f172a', 'transition': 'all 0.3s ease', 'display': 'flex',
        'flexDirection': 'column', 'margin': '0', 'padding': '8px', 'height': 'calc(100vh - 60px)',
        'boxSizing': 'border-box', 'overflow': 'hidden', 'gap': '8px', 'flex': '1'
    }, children=[
    # MAIN MAP + SIDEBAR ROW
    html.Div(id='main-content', style={
        'display': 'flex', 'height': '65vh', 'margin': '0', 'padding': '0', 'gap': '8px',
        'background': '#0f172a', 'transition': 'all 0.3s ease', 'overflow': 'hidden',
        'boxSizing': 'border-box', 'flex': '0 0 auto'
    }, children=[
        # MAP
        html.Div([
            dcc.Graph(id='main-map', style={'height': '100%', 'width': '100%', 'margin': '0', 'padding': '0'}, config={
                'displayModeBar': False, 'scrollZoom': True,
                'doubleClick': 'reset', 'responsive': False
            })
        ], style={'width': '70%', 'transition': 'all 0.3s ease', 'height': '100%', 'margin': '0', 'padding': '0',
                 'borderRadius': '8px', 'border': '1px solid #334155', 'boxSizing': 'border-box', 'background': '#1e293b', 'overflow': 'hidden'}),

        # SIDEBAR
        html.Div(id='sidebar', style={
            'width': '30%', 'height': '100%', 'overflowY': 'auto', 'overflowX': 'hidden',
            'transition': 'all 0.3s ease', 'margin': '0', 'padding': '12px', 'boxSizing': 'border-box',
            'background': '#1e293b', 'border': '1px solid #334155', 'borderRadius': '8px'
        }, children=[
            # Zone Selector
            html.Div([
                html.Label('Select zone to inspect', id='zone-selector-label', style={
                    'fontSize': '11px', 'textTransform': 'uppercase',
                    'letterSpacing': '1.2px', 'marginBottom': '10px', 'display': 'block',
                    'fontWeight': '600'
                }),
                dcc.Dropdown(
                    id='zone-selector', clearable=False,
                    options=[{'label': z['name'], 'value': z['name']} for z in DATA['zones']],
                    value=DATA['zones'][0]['name'],
                    style={'fontSize': '13px', 'borderRadius': '6px'}
                ),
            ], id='zone-selector-card', style={
                'background': '#1e293b', 'borderRadius': '8px', 'padding': '12px',
                'marginBottom': '12px', 'border': '1px solid #334155', 'margin': '0 0 12px 0',
                'transition': 'all 0.3s ease', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),

            # Precursor Section
            html.Div(id='precursor-section', style={
                'background': '#1e293b', 'borderRadius': '8px', 'padding': '12px',
                'marginBottom': '12px', 'border': '1px solid #334155', 'margin': '0 0 12px 0',
                'transition': 'all 0.3s ease', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),

            # Active Zones
            html.Div([
                html.H4('Active Threat Zones', id='active-zones-heading', style={
                    'margin': '0 0 10px 0', 'fontSize': '12px',
                    'textTransform': 'uppercase', 'letterSpacing': '1.2px', 'fontWeight': '700'
                }),
                html.Div(id='zones-list'),
            ], id='active-zones-card', style={
                'background': '#1e293b', 'borderRadius': '8px', 'padding': '12px',
                'border': '1px solid #334155', 'transition': 'all 0.3s ease',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin': '0'
            }),
        ]),
    ]),

    # ─────────────────────────────────────────────
    # RUNOFF → HYPOXIA FLOW PATHWAYS PANEL
    # ─────────────────────────────────────────────
    html.Div([
        # Header
        html.Div([
            html.H2('🌊 Runoff → Hypoxia Flow Pathways', id='flow-panel-heading', style={
                'margin': '0', 'fontSize': '16px', 'fontWeight': '700',
                'letterSpacing': '0.5px', 'padding': '0', 'color': '#f1f5f9'
            }),
            html.P('Visualizing nutrient transport from agricultural runoff to dead zone formation',
                style={
                    'margin': '4px 0 0 0', 'fontSize': '11px',
                    'letterSpacing': '0.3px', 'padding': '0', 'color': '#cbd5e1'
                }, id='flow-panel-subtitle')
        ], style={'marginBottom': '0', 'margin-top': '0', 'padding': '8px', 'background': '#1e293b',
                 'borderRadius': '6px', 'border': '1px solid #475569', 'boxSizing': 'border-box',
                 'flex': '0 0 auto'}),

        # Flow Visualization Map
        html.Div([
            dcc.Graph(
                id='flow-pathways-map',
                style={'height': '100%', 'width': '100%', 'margin': '0', 'padding': '0'},
                config={'displayModeBar': True, 'responsive': True, 'scrollZoom': True}
            ),
            html.Div(id='flow-panel-error', style={
                'display': 'none', 'padding': '12px', 'borderRadius': '6px',
                'fontSize': '12px', 'marginTop': '8px', 'background': '#dc2626', 'color': '#fff'
            })
        ], style={'height': '100%', 'margin': '0', 'padding': '0', 'boxSizing': 'border-box',
                 'border': '1px solid #475569', 'borderRadius': '6px', 'background': '#1e293b',
                 'flex': '1 1 auto', 'minHeight': '0', 'overflow': 'hidden', 'position': 'relative'}),

        # Flow Legend
        html.Div(id='flow-legend', style={
            'marginTop': '0', 'display': 'flex', 'gap': '12px', 'fontSize': '10px',
            'margin': '0', 'padding': '8px', 'boxSizing': 'border-box',
            'border': '1px solid #475569', 'borderRadius': '4px', 'background': '#1e293b',
            'flex': '0 0 auto', 'zIndex': '10', 'position': 'relative', 'flexWrap': 'wrap'
        }, children=[
            html.Div([
                html.Div(style={'width': '12px', 'height': '12px', 'background': '#fbbf24', 'borderRadius': '2px'}),
                'Weak (0-33%)'
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px', 'color': '#f1f5f9'}),
            html.Div([
                html.Div(style={'width': '12px', 'height': '12px', 'background': '#f97316', 'borderRadius': '2px'}),
                'Medium (33-66%)'
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px', 'color': '#f1f5f9'}),
            html.Div([
                html.Div(style={'width': '12px', 'height': '12px', 'background': '#ef4444', 'borderRadius': '2px'}),
                'Strong (66-100%)'
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px', 'color': '#f1f5f9'}),
        ]),

        # Data Store for caching
        dcc.Store(id='flow-pathways-store', data=None),

    ], id='flow-panel', style={
        'marginTop': '0', 'height': '30vh', 'padding': '12px', 'background': '#1e293b', 
        'borderRadius': '8px', 'border': '2px solid #334155', 'margin': '0',
        'transition': 'all 0.3s ease', 'overflow': 'hidden', 'boxSizing': 'border-box',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.15)', 'borderTop': '2px solid #475569',
        'display': 'flex', 'flexDirection': 'column', 'gap': '8px'
    }),
    ]),
])

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────

@callback(
    Output('flow-pathways-store', 'data'),
    Input('data-store', 'data'),
    prevent_initial_call=False
)
def fetch_flow_pathways(data):
    """Fetch runoff-to-hypoxia pathways from backend and cache."""
    try:
        logger.info("🔄 Fetching flow pathways from /api/runoff-to-hypoxia-pathways...")
        response = requests.get(
            "http://localhost:5000/api/runoff-to-hypoxia-pathways",
            timeout=5
        )
        response.raise_for_status()
        api_response = response.json()
        
        # API returns 'links' directly - pass through as-is
        pathways_data = api_response
        logger.info(f"✅ Flow pathways fetched successfully. Links: {len(pathways_data.get('links', []))}")
        return pathways_data
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Connection failed: Backend not running? {e}")
        return {"links": []}
    except Exception as e:
        logger.error(f"❌ Flow pathways fetch failed: {e}")
        return {"links": []}


@callback(
    [Output('flow-pathways-map', 'figure'),
     Output('flow-panel-error', 'style')],
    [Input('flow-pathways-store', 'data'),
     Input('theme-store', 'data')],
    prevent_initial_call=False
)
def render_flow_pathways(pathways_data, theme):
    """
    Render the runoff-to-hypoxia flow visualization.
    
    Visualization logic:
    1. Flow lines connect rivers → ocean transport → dead zones
    2. Color represents correlation strength (nutrient impact)
    3. Opacity and width scale with correlation strength
    4. Start with green markers (nutrient sources)
    5. End with red markers (hypoxia zones)
    """
    colors = get_color_scheme(theme)
    fig = go.Figure()
    
    # ═══════════════════════════════════════════════════════════
    # DEBUG LOGGING
    # ═══════════════════════════════════════════════════════════
    logger.info(f"📊 render_flow_pathways called with data: {type(pathways_data)}")
    if pathways_data:
        logger.info(f"   Data keys: {pathways_data.keys()}")
        if 'links' in pathways_data:
            logger.info(f"   Number of links: {len(pathways_data['links'])}")
    
    # ═══════════════════════════════════════════════════════════
    # VALIDATION - Check for empty or malformed data
    # ═══════════════════════════════════════════════════════════
    if not pathways_data or not isinstance(pathways_data, dict):
        logger.warning("❌ Invalid pathways data (None or not a dict)")
        fig.add_annotation(
            text="No runoff flow data available",
            showarrow=False,
            font=dict(size=14, color=colors['text_muted']),
            xref="paper", yref="paper", x=0.5, y=0.5
        )
        error_style = {'display': 'block', 'padding': '16px', 'background': colors['red'],
                      'border': f'1px solid {colors["red"]}', 'borderRadius': '8px',
                      'color': colors['card'], 'fontSize': '13px', 'marginTop': '12px', 'opacity': '0.8'}
        return fig, error_style, []
    
    links = pathways_data.get('links', [])
    if not links or len(links) == 0:
        logger.warning("⚠️ No links in pathways data")
        fig.add_annotation(
            text="No runoff-hypoxia pathways detected",
            showarrow=False,
            font=dict(size=14, color=colors['text_muted']),
            xref="paper", yref="paper", x=0.5, y=0.5
        )
        error_style = {'display': 'block', 'padding': '16px', 'background': colors['orange'],
                      'border': f'1px solid {colors["orange"]}', 'borderRadius': '8px',
                      'color': colors['card'], 'fontSize': '13px', 'marginTop': '12px', 'opacity': '0.8'}
        return fig, error_style, []
    
    logger.info(f"✅ Processing {len(links)} pathways")
    error_style = {'display': 'none'}
    stats = []
    total_strength = 0
    pathway_count = 0
    valid_pathways = 0
    
    # Process each pathway
    for idx, pathway in enumerate(links):
        try:
            # Validate pathway structure
            if not isinstance(pathway, dict):
                logger.warning(f"⚠️ Pathway {idx} is not a dict: {type(pathway)}")
                continue
            
            plume_path = pathway.get('plume_path', [])
            if not plume_path:
                logger.warning(f"⚠️ Pathway {idx} has no plume_path")
                continue
            
            if not isinstance(plume_path, list):
                logger.warning(f"⚠️ Pathway {idx} plume_path is not a list: {type(plume_path)}")
                continue
            
            if len(plume_path) < 2:
                logger.warning(f"⚠️ Pathway {idx} plume_path has < 2 points: {len(plume_path)}")
                continue
            
            logger.info(f"✅ Processing pathway {idx}: {len(plume_path)} points")
            
            pathway_count += 1
            correlation_strength = pathway.get('correlation_strength', 0.5)
            if not isinstance(correlation_strength, (int, float)):
                correlation_strength = 0.5
            correlation_strength = max(0, min(1, correlation_strength))  # Clamp to 0-1
            
            total_strength += correlation_strength
            river_name = str(pathway.get('river_name', f'River {idx}'))
            downstream_zone = str(pathway.get('downstream_zone', 'Dead Zone'))
            
            # Extract path coordinates - CRITICAL FIX
            plume_lats = []
            plume_lons = []
            for point_idx, point in enumerate(plume_path):
                try:
                    if isinstance(point, dict):
                        # Handle dict format: {'lat': ..., 'lon': ...}
                        lat = point.get('lat', None)
                        lon = point.get('lon', None)
                        if lat is not None and lon is not None:
                            plume_lats.append(float(lat))
                            plume_lons.append(float(lon))
                    elif isinstance(point, (list, tuple)) and len(point) >= 2:
                        # Handle list/tuple format: [lat, lon] or (lat, lon)
                        plume_lats.append(float(point[0]))
                        plume_lons.append(float(point[1]))
                except (ValueError, TypeError) as pe:
                    logger.warning(f"⚠️ Could not parse point {point_idx} in pathway {idx}: {point}")
                    continue
            
            if len(plume_lats) < 2:
                logger.warning(f"⚠️ Pathway {idx} has {len(plume_lats)} valid points (need ≥2)")
                continue
            
            logger.info(f"✅ Pathway {idx} extracted {len(plume_lats)} coordinates")
            valid_pathways += 1
            
            # Determine color based on correlation strength (weak→yellow, medium→orange, strong→red)
            if correlation_strength > 0.7:
                color = '#dc2626'  # Deep red - strong nutrient impact
                marker_color = '#991b1b'
            elif correlation_strength > 0.4:
                color = '#f97316'  # Orange - medium impact
                marker_color = '#ea580c'
            else:
                color = '#fbbf24'  # Yellow - weak impact
                marker_color = '#f59e0b'
            
            # Opacity proportional to strength (0.4 to 1.0)
            opacity = 0.4 + (correlation_strength * 0.6)
            
            # Line width proportional to strength (1 to 4)
            line_width = 1 + (correlation_strength * 3)
            
            # ═══════════════════════════════════════════════════════════
            # FLOW LINE - Main plume trajectory
            # ═══════════════════════════════════════════════════════════
            fig.add_trace(go.Scattermapbox(
                lat=plume_lats, lon=plume_lons,
                mode='lines',
                line=dict(color=color, width=line_width),
                opacity=opacity,
                name=f"Transport: {river_name} → {downstream_zone}",
                hovertemplate=(
                    f"<b>🌊 Nutrient Transport Path</b><br>"
                    f"River: {river_name}<br>"
                    f"Target Zone: {downstream_zone}<br>"
                    f"Correlation Strength: {correlation_strength:.0%}<br>"
                    f"<i>Nutrient runoff contributing to hypoxia</i>"
                    f"<extra></extra>"
                ),
                showlegend=True
            ))
            
            # ═══════════════════════════════════════════════════════════
            # FLOW DIRECTION MARKERS - Show movement along path
            # ═══════════════════════════════════════════════════════════
            if len(plume_lats) > 3:
                # Add intermediate markers every 3rd point to show flow direction
                marker_indices = list(range(0, len(plume_lats), max(1, len(plume_lats) // 4)))
                if len(plume_lats) - 1 not in marker_indices:
                    marker_indices.append(len(plume_lats) - 1)
                
                marker_lats = [plume_lats[i] for i in marker_indices[1:-1]]  # Exclude start/end
                marker_lons = [plume_lons[i] for i in marker_indices[1:-1]]
                
                if marker_lats:
                    fig.add_trace(go.Scattermapbox(
                        lat=marker_lats, lon=marker_lons,
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=marker_color,
                            opacity=0.7,
                            symbol='circle'
                        ),
                        showlegend=False,
                        hovertemplate=f"<b>Flow checkpoint</b><br>{river_name} transport<extra></extra>",
                        name='Flow Direction'
                    ))
            
            # ═══════════════════════════════════════════════════════════
            # RIVER SOURCE POINT - Green marker at start
            # ═══════════════════════════════════════════════════════════
            fig.add_trace(go.Scattermapbox(
                lat=[plume_lats[0]], lon=[plume_lons[0]],
                mode='markers',
                marker=dict(
                    size=14,
                    color='#15803d',  # Green - nutrient source
                    opacity=0.9,
                    symbol='circle'
                ),
                showlegend=True,
                name=f"🟢 River Source: {river_name}",
                hovertemplate=(
                    f"<b>🌱 Nutrient Source</b><br>"
                    f"River: {river_name}<br>"
                    f"Initiates runoff transport<extra></extra>"
                )
            ))
            
            # ═══════════════════════════════════════════════════════════
            # DEAD ZONE ENDPOINT - Red glowing marker
            # ═══════════════════════════════════════════════════════════
            fig.add_trace(go.Scattermapbox(
                lat=[plume_lats[-1]], lon=[plume_lons[-1]],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='#dc2626',  # Red - hypoxia zone
                    opacity=0.95,
                    symbol='star'
                ),
                text=['⚠️'],
                textposition='middle center',
                textfont=dict(size=12),
                showlegend=True,
                name=f"🔴 Dead Zone: {downstream_zone}",
                hovertemplate=(
                    f"<b>☠️ Hypoxic Dead Zone</b><br>"
                    f"Zone: {downstream_zone}<br>"
                    f"Nutrient Impact: {correlation_strength:.0%}<br>"
                    f"<i>Result of nutrient accumulation</i><extra></extra>"
                )
            ))
            
        except Exception as e:
            logger.error(f"❌ Error processing pathway {idx}: {e}")
            continue
    
    logger.info(f"📊 Valid pathways rendered: {valid_pathways}/{pathway_count}")
    
    # ═══════════════════════════════════════════════════════════
    # MAP LAYOUT CONFIGURATION - CRITICAL FOR PROPER RENDERING
    # ═══════════════════════════════════════════════════════════
    fig.update_layout(
        mapbox=dict(
            style=colors['map_style'],
            center=dict(lat=15, lon=80),
            zoom=4,
            bearing=0,
            pitch=0
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        dragmode='pan',
        uirevision='constant',
        legend=dict(
            bgcolor=rgba_str(colors['card'], 0.85),
            font=dict(color=colors['text'], size=9),
            x=0.01, y=0.95,
            bordercolor=colors['border'],
            borderwidth=1,
            yanchor='top'
        ),
        showlegend=True,
        autosize=True,
        height=350
    )
    
    # ═══════════════════════════════════════════════════════════
    # FLOW STATISTICS - Summary cards
    # ═══════════════════════════════════════════════════════════
    if valid_pathways > 0:
        avg_strength = total_strength / pathway_count
        
        stats = [
            html.Div([
                html.Div('🔗 Active Pathways', style={
                    'fontSize': '12px', 'fontWeight': '600', 'marginBottom': '6px',
                    'color': colors['text_muted'], 'textTransform': 'uppercase'
                }),
                html.Div(str(valid_pathways), style={
                    'fontSize': '24px', 'fontWeight': '700', 'color': colors['blue']
                })
            ], style=_stat_card_style(colors)),
            
            html.Div([
                html.Div('📊 Avg Correlation', style={
                    'fontSize': '12px', 'fontWeight': '600', 'marginBottom': '6px',
                    'color': colors['text_muted'], 'textTransform': 'uppercase'
                }),
                html.Div(f"{avg_strength:.0%}", style={
                    'fontSize': '24px', 'fontWeight': '700', 'color': colors['orange']
                })
            ], style=_stat_card_style(colors)),
            
            html.Div([
                html.Div('🎯 High Impact Links', style={
                    'fontSize': '12px', 'fontWeight': '600', 'marginBottom': '6px',
                    'color': colors['text_muted'], 'textTransform': 'uppercase'
                }),
                html.Div(
                    str(sum(1 for p in links if p.get('correlation_strength', 0) > 0.7)),
                    style={'fontSize': '24px', 'fontWeight': '700', 'color': colors['red']}
                )
            ], style=_stat_card_style(colors)),
        ]
    else:
        logger.warning("⚠️ No valid pathways to display")
    
    return fig, error_style


def _stat_card_style(colors):
    """Helper function to generate stat card styling."""
    return {
        'padding': '14px', 'background': colors['bg'],
        'border': f'1px solid {colors["border"]}', 'borderRadius': '8px',
        'transition': 'all 0.3s ease'
    }


@callback(
    [Output('theme-store', 'data'),
     Output('theme-toggle', 'children')],
    Input('theme-toggle', 'n_clicks'),
    State('theme-store', 'data'),
    prevent_initial_call=False
)
def toggle_theme(n_clicks, current_theme):
    new_theme = 'light' if current_theme == 'dark' else 'dark'
    icon = '☀️' if new_theme == 'dark' else '🌙'
    return new_theme, icon


@callback(
    [Output('main-container', 'style'),
     Output('navbar', 'style'),
     Output('content-wrapper', 'style'),
     Output('main-content', 'style'),
     Output('sidebar', 'style'),
     Output('precursor-section', 'style'),
     Output('zone-selector-card', 'style'),
     Output('active-zones-card', 'style'),
     Output('layer-toggle', 'style'),
     Output('navbar-subtitle', 'style'),
     Output('theme-toggle', 'style')],
    Input('theme-store', 'data')
)
def update_theme_styles(theme):
    colors = get_color_scheme(theme)
    
    container_style = {
        'background': colors['bg'], 'minHeight': '100vh', 'fontFamily': 'Inter, system-ui, sans-serif',
        'color': colors['text'], 'transition': 'background 0.3s ease, color 0.3s ease'
    }
    
    navbar_style = {
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
        'padding': '8px 24px', 'borderBottom': f'1px solid {colors["border"]}',
        'background': colors['bg'], 'transition': 'all 0.3s ease'
    }
    
    wrapper_style = {
        'background': colors['bg'], 'transition': 'all 0.3s ease'
    }
    
    content_style = {
        'display': 'flex', 'padding': '0', 'margin': '0', 'gap': '0',
        'background': colors['bg'], 'transition': 'all 0.3s ease',
        'height': '65vh', 'boxSizing': 'border-box'
    }
    
    sidebar_style = {
        'width': '30%', 'overflowY': 'auto', 'maxHeight': '65vh',
        'transition': 'all 0.3s ease', 'height': '65vh', 'padding': '12px',
        'boxSizing': 'border-box', 'background': colors['bg']
    }
    
    card_style = {
        'background': colors['card'], 'borderRadius': '8px', 'padding': '12px',
        'marginBottom': '12px', 'border': f'1px solid {colors["border"]}',
        'transition': 'all 0.3s ease', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }
    
    toggle_style = {
        'color': colors['text'], 'fontSize': '12px', 'display': 'flex',
        'gap': '2px', 'marginRight': '24px', 'transition': 'color 0.3s ease'
    }
    
    subtitle_style = {
        'color': colors['text_muted'], 'fontSize': '12px', 'marginTop': '2px'
    }
    
    theme_toggle_style = {
        'background': colors['card'], 'border': f'1px solid {colors["border"]}', 'color': colors['text'],
        'fontSize': '18px', 'padding': '8px 14px', 'borderRadius': '8px',
        'cursor': 'pointer', 'transition': 'all 0.3s ease',
        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
    }
    
    return (container_style, navbar_style, wrapper_style, content_style, sidebar_style,
            card_style, card_style, card_style, toggle_style, subtitle_style, theme_toggle_style)


@callback(
    [Output('zone-selector-label', 'style'),
     Output('active-zones-heading', 'style'),
     Output('anoxia-title', 'style'),
     Output('flow-panel', 'style'),
     Output('flow-panel-heading', 'style'),
     Output('flow-panel-subtitle', 'style')],
    Input('theme-store', 'data'),
    prevent_initial_call=False
)
def update_text_colors(theme):
    colors = get_color_scheme(theme)
    
    label_style = {
        'color': colors['text_muted'], 'fontSize': '11px', 'textTransform': 'uppercase',
        'letterSpacing': '1.2px', 'marginBottom': '10px', 'display': 'block',
        'fontWeight': '600'
    }
    
    heading_style = {
        'color': colors['text'], 'margin': '0 0 14px 0', 'fontSize': '13px',
        'textTransform': 'uppercase', 'letterSpacing': '1.2px', 'fontWeight': '700'
    }
    
    title_style = {
        'color': colors['text'], 'margin': '0', 'fontSize': '28px',
        'letterSpacing': '4px', 'fontWeight': '800'
    }
    
    flow_panel_style = {
        'padding': '0', 'margin': '0', 'background': colors['card'], 'borderRadius': '0',
        'border': f'1px solid {colors["border"]}', 'transition': 'all 0.3s ease'
    }
    
    flow_heading_style = {
        'margin': '0', 'fontSize': '18px', 'fontWeight': '700',
        'letterSpacing': '0.5px', 'color': colors['text']
    }
    
    flow_subtitle_style = {
        'margin': '4px 0 0 0', 'fontSize': '12px',
        'letterSpacing': '0.3px', 'color': colors['text_muted']
    }
    
    return (label_style, heading_style, title_style, flow_panel_style, 
            flow_heading_style, flow_subtitle_style)


@callback(
    Output('main-map', 'figure'),
    [Input('layer-toggle', 'value'),
     Input('data-store', 'data'),
     Input('theme-store', 'data'),
     Input('wind-vectors-store', 'data'),
     Input('ocean-currents-store', 'data')]
)
def update_map(layer, data, theme, wind_data, currents_data):
    colors = get_color_scheme(theme)
    fig = go.Figure()

    # Dead Zone Heatmap - OCEAN ONLY (Arabian Sea + Bay of Bengal)
    if layer in ['all', 'dz']:
        dz_probs = data.get('dz_probs', [])
        if dz_probs:
            dz_probs = np.array(dz_probs)
            dz_lats = np.array(data.get('dz_lats', []))
            dz_lons = np.array(data.get('dz_lons', []))
            
            # CRITICAL: Filter to ONLY ocean regions (Arabian Sea + Bay of Bengal)
            # Arabian Sea: 8-20°N, 50-75°E
            # Bay of Bengal: 8-20°N, 85-100°E
            arabian_sea_mask = (dz_lats >= 8) & (dz_lats <= 20) & (dz_lons >= 50) & (dz_lons <= 75)
            bay_of_bengal_mask = (dz_lats >= 8) & (dz_lats <= 20) & (dz_lons >= 85) & (dz_lons <= 100)
            ocean_mask = arabian_sea_mask | bay_of_bengal_mask
            
            # Filter to ONLY high-confidence ocean dead zones (>= 0.20 probability)
            high_prob_mask = dz_probs >= 0.20
            combined_mask = ocean_mask & high_prob_mask
            
            filtered_lats = dz_lats[combined_mask]
            filtered_lons = dz_lons[combined_mask]
            filtered_probs = dz_probs[combined_mask]
            
            if len(filtered_probs) > 0:
                # Use Densitymapbox for proper heatmap (not scattered dots)
                fig.add_trace(go.Densitymapbox(
                    lat=filtered_lats, lon=filtered_lons,
                    z=filtered_probs, radius=50,
                    colorscale=[
                        [0.0, 'rgba(59, 130, 246, 0.1)'],       # Light blue for 20%
                        [0.3, 'rgba(251, 191, 36, 0.6)'],       # Yellow for 40%
                        [0.6, 'rgba(249, 115, 22, 0.75)'],      # Orange for 60%
                        [1.0, 'rgba(239, 68, 68, 0.95)']        # Red for 100% (true dead zones)
                    ],
                    zmin=0.2, zmax=1.0,
                    opacity=0.8, showscale=True, name='Dead Zone Risk',
                    colorbar=dict(
                        thickness=15, len=0.7, x=1.02,
                        tickfont=dict(size=10),
                        title=dict(text='P(Hypoxia)<br>20-100%', font=dict(size=11)),
                        tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                        ticktext=['20%', '40%', '60%', '80%', '100%']
                    ),
                    hovertemplate='<b>Dead Zone</b><br>P(hypoxia): %{z:.1%}<extra></extra>',
                    customdata=list(zip(filtered_lats, filtered_lons))
                ))

    # Ghost Gear Paths
    if layer in ['all', 'gg']:
        for path in data.get('drift_paths', []):
            fig.add_trace(go.Scattermapbox(
                lat=path['lats'], lon=path['lons'], mode='lines+markers',
                line=dict(color=colors['purple'], width=2.5),
                marker=dict(size=6, color=colors['purple']),
                name=f"Ghost Gear Path {path['id']}",
                hovertemplate=f"<b>Ghost Gear {path['id']}</b><extra></extra>",
            ))

    # Biodiversity Traps
    if layer == 'all' and data.get('traps'):
        for trap in data.get('traps', []):
            fig.add_trace(go.Scattermapbox(
                lat=[trap['lat']], lon=[trap['lon']], mode='markers',
                marker=dict(size=18, color=colors['orange'], opacity=0.95),
                name='Biodiversity Trap',
                hovertemplate=f"<b>⚠️ Trap</b><br>P(hypoxia): {trap['p_hypoxia']:.0%}<extra></extra>",
            ))

    # WIND VECTORS OVERLAY (with directional arrows)
    if wind_data and 'vectors' in wind_data:
        for vector in wind_data['vectors']:
            lat = vector['lat']
            lon = vector['lon']
            u = vector['u'] * 2
            v = vector['v'] * 2
            mag = vector['magnitude']
            arrow_color = colors['red'] if mag > 0.25 else (colors['orange'] if mag > 0.15 else colors['blue'])
            
            # Main vector line
            fig.add_trace(go.Scattermapbox(
                lat=[lat, lat + v], lon=[lon, lon + u],
                mode='lines', line=dict(color=arrow_color, width=2.5),
                showlegend=False, hovertemplate=f"<b>Wind</b><br>Speed: {mag:.2f} m/s<extra></extra>",
                name='Wind Vector'
            ))
            
            # Arrow head at endpoint
            fig.add_trace(go.Scattermapbox(
                lat=[lat + v], lon=[lon + u],
                mode='markers',
                marker=dict(size=8, color=arrow_color, symbol='arrow', angle=0),
                showlegend=False,
                hovertemplate=f"<b>Wind Direction</b><br>Speed: {mag:.2f} m/s<extra></extra>"
            ))
    
    # OCEAN CURRENTS OVERLAY (with directional arrows)
    if currents_data and 'vectors' in currents_data:
        for vector in currents_data['vectors']:
            lat = vector['lat']
            lon = vector['lon']
            u = vector['u'] * 1.5
            v = vector['v'] * 1.5
            
            # Main vector line
            fig.add_trace(go.Scattermapbox(
                lat=[lat, lat + v], lon=[lon, lon + u],
                mode='lines', line=dict(color=colors['purple'], width=2.5),
                showlegend=False, hovertemplate=f"<b>Current</b><br>Speed: {vector['magnitude']:.2f} cm/s<extra></extra>",
                name='Ocean Current'
            ))
            
            # Arrow head at endpoint
            fig.add_trace(go.Scattermapbox(
                lat=[lat + v], lon=[lon + u],
                mode='markers',
                marker=dict(size=8, color=colors['purple'], symbol='arrow', angle=0),
                showlegend=False,
                hovertemplate=f"<b>Current Direction</b><br>Speed: {vector['magnitude']:.2f} cm/s<extra></extra>"
            ))
    
    fig.update_layout(
        mapbox=dict(
            style=colors['map_style'],
            center=dict(lat=13.5, lon=75),
            zoom=4.5,
        ),
        uirevision='constant',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor=rgba_str(colors['card'], 0.85),
            font=dict(color=colors['text'], size=11),
            x=0.01, y=0.99, 
            bordercolor=colors['border'], 
            borderwidth=1
        ),
        showlegend=True,
        dragmode='pan',
        hovermode='closest',
    )
    return fig


@callback(
    Output('zones-list', 'children'),
    [Input('data-store', 'data'),
     Input('theme-store', 'data')]
)
def update_zones_list(data, theme):
    return [zone_card(z, theme) for z in data.get('zones', [])]


def calculate_precursor_conditions(lat, lon, probability):
    """
    Calculate dynamic precursor conditions based on location and probability.
    Uses backend probability to infer environmental conditions.
    """
    # Normalize lat/lon to 0-1 range for calculations
    norm_lat = (lat - 8) / 12  # 8-20°N range
    norm_lon = (lon - 50) / 50  # 50-100°E range
    
    # Clamp to 0-1
    norm_lat = max(0, min(1, norm_lat))
    norm_lon = max(0, min(1, norm_lon))
    
    # Probability drives severity of all parameters
    prob_factor = probability * 100  # Scale to 0-100
    
    # NITRATE ANOMALY: Higher in Bay of Bengal (east), increases with probability
    # Bay of Bengal tends to have higher nutrient cycling
    bay_bengal_factor = 1.2 if norm_lon > 0.7 else 0.9
    nitrate = int(150 + (prob_factor * 1.8 * bay_bengal_factor))
    
    # CHLOROPHYLL: High in productive coastal zones, correlates with probability
    chlor = int(80 + (prob_factor * 2.2 * (0.8 + norm_lat * 0.4)))
    
    # THERMAL STRATIFICATION: Varies by region
    # Bay of Bengal: MEDIUM, Arabian Sea: HIGH
    if norm_lon > 0.7:
        therm_val = "STRONG" if prob_factor > 70 else "MEDIUM" if prob_factor > 40 else "WEAK"
        therm_pct = min(95, int(50 + prob_factor * 0.6))
    else:
        therm_val = "HIGH" if prob_factor > 60 else "MEDIUM" if prob_factor > 30 else "LOW"
        therm_pct = min(95, int(70 + prob_factor * 0.5))
    
    # WIND STRESS MIXING: Inverse relationship with probability
    # Low wind = hypoxia develops, high wind = mixing prevents it
    wind_pct = max(15, int(80 - prob_factor * 0.9))
    wind_val = "LOW" if wind_pct < 40 else "MEDIUM" if wind_pct < 60 else "HIGH"
    
    # DO DRAWDOWN RATE: Higher = faster depletion = more hypoxia
    do_rate = "VERY FAST" if prob_factor > 80 else "FAST" if prob_factor > 50 else "MODERATE"
    do_pct = min(95, int(30 + prob_factor * 1.0))
    
    return [
        ("Nitrate anomaly",        f"+{nitrate}%", min(95, int(prob_factor * 1.2))),
        ("Chlorophyl-a (MODIS)",   f"+{chlor}%", chlor),
        ("Thermal stratification", therm_val,  therm_pct),
        ("Wind stress mixing",     wind_val,   wind_pct),
        ("DO drawdown rate",       do_rate,    do_pct),
    ]


def precursor_bar(label, value, pct, theme):
    """Render a precursor condition bar."""
    colors = get_color_scheme(theme)
    if pct >= 80:
        bar_color = colors['red']
    elif pct >= 50:
        bar_color = colors['orange']
    elif pct >= 30:
        bar_color = colors['blue']
    else:
        bar_color = colors['green']
    
    return html.Div([
        html.Div([
            html.Span(label, style={'color': colors['text'], 'fontSize': '12px', 'flex': '1'}),
            html.Span(value, style={'color': bar_color, 'fontSize': '12px', 'fontWeight': '600',
                'marginLeft': '10px' if isinstance(value, str) and value[0].isalpha() else '0'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '8px'}),
        html.Div(style={'background': f'rgba(0,0,0,0.2)', 'height': '6px', 'borderRadius': '3px',
            'overflow': 'hidden', 'marginBottom': '10px'}, children=[
            html.Div(style={'background': bar_color, 'width': f'{pct}%', 'height': '100%',
                'borderRadius': '3px', 'transition': 'width 0.3s ease'})
        ])
    ])


@callback(
    Output('map-click-store', 'data'),
    Input('main-map', 'clickData')
)
def capture_map_click(clickData):
    if clickData is None or 'points' not in clickData or len(clickData['points']) == 0:
        return None
    point = clickData['points'][0]
    return {'lat': point.get('lat'), 'lon': point.get('lon')}


@callback(
    Output('precursor-section', 'children'),
    [Input('zone-selector', 'value'),
     Input('map-click-store', 'data'),
     Input('theme-store', 'data')],
    prevent_initial_call=False
)
def update_precursor_enhanced(zone_name, click_data, theme):
    colors = get_color_scheme(theme)
    interventions = []
    bars = []
    location_info = ""

    # ═══════════════════════════════════════════════════════════
    # DEBUG LOGGING
    # ═══════════════════════════════════════════════════════════
    logger.info(f"📍 Precursor update triggered")
    logger.info(f"   Zone: {zone_name}")
    logger.info(f"   Click data: {click_data}")

    # ─────────────────────────────
    # MAP CLICK → USE API
    # ─────────────────────────────
    if click_data and isinstance(click_data, dict):
        click_lat = click_data.get('lat')
        click_lon = click_data.get('lon')
        
        logger.info(f"   Map click detected: lat={click_lat}, lon={click_lon}")
        
        if click_lat is not None and click_lon is not None:
            try:
                click_lat = float(click_lat)
                click_lon = float(click_lon)
                
                logger.info(f"🔄 Fetching precursor conditions from API...")
                response = requests.get(
                    f"http://localhost:5000/api/precursor-conditions/{click_lat}/{click_lon}",
                    timeout=5
                )
                response.raise_for_status()
                
                api_data = response.json()
                logger.info(f"✅ API response keys: {api_data.keys()}")
                
                # Try multiple possible keys for precursor data
                precursor = api_data.get('precursors', {}) or api_data.get('precursor_conditions', {})
                logger.info(f"   Precursor data: {precursor}")

                bars = [
                    ("Nitrate anomaly", f"+{precursor.get('nitrate_anomaly', 0):.0f}%", min(100, max(0, precursor.get('nitrate_anomaly', 0)))),
                    ("Chlorophyll-a (MODIS)", f"+{precursor.get('chlorophyll_modis', 0):.0f}%", min(100, max(0, precursor.get('chlorophyll_modis', 0)))),
                    ("Thermal stratification", precursor.get('thermal_stratification', 'UNKNOWN'), precursor.get('thermal_stratification_pct', 0)),
                    ("Wind stress mixing", precursor.get('wind_stress', 'UNKNOWN'), precursor.get('wind_stress_pct', 0)),
                    ("DO drawdown rate", precursor.get('do_drawdown', 'UNKNOWN'), precursor.get('do_drawdown_pct', 0)),
                ]
                
                location_info = f" @ ({click_lat:.2f}°N, {click_lon:.2f}°E)"
                logger.info(f"✅ Bars populated from API: {len(bars)} items")
                
                # FETCH INTERVENTION MEASURES
                try:
                    logger.info(f"🔄 Fetching interventions...")
                    interv_response = requests.get(
                        f"http://localhost:5000/api/intervention-measures/{click_lat}/{click_lon}",
                        timeout=5
                    )
                    interv_response.raise_for_status()
                    interventions = interv_response.json().get('interventions', [])
                    logger.info(f"✅ Interventions fetched: {len(interventions)} items")
                except Exception as e:
                    logger.error(f"❌ Intervention fetch failed: {e}")
                    interventions = []

            except ValueError as ve:
                logger.error(f"❌ Invalid lat/lon values: {click_lat}, {click_lon}")
                bars = PRECURSOR_DATA.get(zone_name, [])
            except requests.exceptions.ConnectionError as ce:
                logger.error(f"❌ Connection error (backend running?): {ce}")
                bars = PRECURSOR_DATA.get(zone_name, [])
            except Exception as e:
                logger.error(f"❌ API error: {e}")
                bars = PRECURSOR_DATA.get(zone_name, [])
        else:
            logger.warning("⚠️ Click data missing lat/lon")
            bars = PRECURSOR_DATA.get(zone_name, [])
    else:
        logger.info("ℹ️ No map click, using zone selector fallback")
        bars = PRECURSOR_DATA.get(zone_name, [])
        location_info = ""

    # ─────────────────────────────
    # UI RENDER
    # ─────────────────────────────
    if not bars:
        logger.warning("⚠️ No bars data available")
        return html.Div([
            html.H4('Precursor Conditions', style={
                'color': colors['text_muted'], 'marginBottom': '16px',
                'fontSize': '13px', 'textTransform': 'uppercase'
            }),
            html.Div('Click on the map to see conditions', style={
                'color': colors['text_muted'], 'fontSize': '12px'
            })
        ])

    critical_count = sum(1 for _, _, pct in bars if pct >= 80)

    if critical_count == 0:
        title_color = colors['green']
    elif critical_count < 3:
        title_color = colors['orange']
    else:
        title_color = colors['red']

    # BUILD INTERVENTION CARDS (Dynamic from API)
    intervention_elements = []
    if interventions:
        intervention_elements.append(
            html.Div([
                html.H4('Interventions', style={
                    'color': colors['text'], 'marginTop': '24px', 'marginBottom': '4px',
                    'fontSize': '13px', 'textTransform': 'uppercase', 'borderTop': f'1px solid {colors["border"]}',
                    'paddingTop': '16px'
                }),
                html.P('Based on real-time environmental conditions', style={
                    'color': colors['text_muted'], 'fontSize': '11px', 'marginTop': '0px',
                    'marginBottom': '12px', 'fontStyle': 'italic'
                })
            ])
        )
        
        for interv in interventions:
            priority = interv.get('priority', 'ROUTINE')
            
            # Priority color and icon mapping
            priority_styling = {
                'CRITICAL': {
                    'color': colors['red'],
                    'bg': 'rgba(239, 68, 68, 0.15)',
                    'border': f'2px solid {colors["red"]}',
                    'icon': '🚨'
                },
                'URGENT': {
                    'color': colors['orange'],
                    'bg': 'rgba(249, 115, 22, 0.15)',
                    'border': f'2px solid {colors["orange"]}',
                    'icon': '⚠️'
                },
                'WARNING': {
                    'color': colors['blue'],
                    'bg': 'rgba(59, 130, 246, 0.15)',
                    'border': f'1px solid {colors["blue"]}',
                    'icon': '⏱️'
                },
                'HIGH': {
                    'color': colors['orange'],
                    'bg': 'rgba(249, 115, 22, 0.1)',
                    'border': f'1px solid {colors["orange"]}',
                    'icon': '→'
                },
                'ROUTINE': {
                    'color': colors['green'],
                    'bg': 'rgba(34, 197, 94, 0.1)',
                    'border': f'1px solid {colors["green"]}',
                    'icon': '✅'
                }
            }
            
            style = priority_styling.get(priority, priority_styling['ROUTINE'])
            
            # Build measures list
            measures_html = []
            for measure in interv.get('measures', []):
                measures_html.append(
                    html.Li(measure, style={
                        'color': colors['text'], 'fontSize': '11px', 'marginBottom': '6px',
                        'lineHeight': '1.4'
                    })
                )
            
            # Build intervention card
            intervention_elements.append(
                html.Div([
                    # Header: Icon + Title + Priority Badge
                    html.Div([
                        html.Span(style['icon'], style={
                            'fontSize': '16px', 'marginRight': '10px',
                            'display': 'inline-block'
                        }),
                        html.Span(interv.get('title', 'Action'), style={
                            'color': style['color'],
                            'fontWeight': '700',
                            'fontSize': '12px',
                            'flex': '1'
                        }),
                        html.Span(priority, style={
                            'color': 'white',
                            'background': style['color'],
                            'padding': '3px 8px',
                            'borderRadius': '4px',
                            'fontSize': '9px',
                            'fontWeight': '700',
                            'textTransform': 'uppercase'
                        })
                    ], style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'marginBottom': '10px',
                        'gap': '8px'
                    }),
                    
                    # Reason
                    html.Div([
                        html.Span('📌 Reason: ', style={
                            'color': colors['text_muted'],
                            'fontSize': '10px',
                            'fontWeight': '700'
                        }),
                        html.Span(interv.get('reason', 'Environmental condition detected'), style={
                            'color': colors['text'],
                            'fontSize': '11px'
                        })
                    ], style={'marginBottom': '8px'}),
                    
                    # Impact
                    html.Div([
                        html.Span('💡 Impact: ', style={
                            'color': colors['text_muted'],
                            'fontSize': '10px',
                            'fontWeight': '700'
                        }),
                        html.Span(interv.get('impact', 'Mitigates hypoxia risk'), style={
                            'color': colors['text'],
                            'fontSize': '11px'
                        })
                    ], style={'marginBottom': '8px'}),
                    
                    # Timeline
                    html.Div([
                        html.Span('⏰ Timeline: ', style={
                            'color': colors['text_muted'],
                            'fontSize': '10px',
                            'fontWeight': '700'
                        }),
                        html.Span(interv.get('timeline', 'N/A'), style={
                            'color': style['color'],
                            'fontSize': '11px',
                            'fontWeight': '600'
                        })
                    ], style={'marginBottom': '10px'}),
                    
                    # Measures
                    html.Div([
                        html.Span('📋 Recommended Actions:', style={
                            'color': colors['text_muted'],
                            'fontSize': '10px',
                            'fontWeight': '700',
                            'display': 'block',
                            'marginBottom': '6px'
                        }),
                        html.Ul(measures_html, style={
                            'margin': '0',
                            'marginLeft': '8px',
                            'paddingLeft': '18px',
                            'color': colors['text']
                        })
                    ])
                ], style={
                    'background': style['bg'],
                    'border': style['border'],
                    'borderRadius': '8px',
                    'padding': '14px',
                    'marginBottom': '12px',
                    'transition': 'all 0.2s ease'
                })
            )

    return html.Div([
        html.H4(f'Precursor Conditions{location_info}', style={
            'color': title_color, 'marginBottom': '16px',
            'fontSize': '13px', 'textTransform': 'uppercase'
        }),
        *[precursor_bar(label, val, pct, theme) for label, val, pct in bars],
        *intervention_elements
    ])


@callback(
    Output('data-store', 'data'),
    Input('data-refresh', 'n_intervals')
)
def refresh_data(n):
    return load_data()


# ─────────────────────────────────────────────
# OVERLAY CALLBACKS (Wind, Currents, Runoff)
# ─────────────────────────────────────────────

@callback(
    Output('wind-vectors-store', 'data'),
    Input('toggle-wind-btn', 'n_clicks'),
    prevent_initial_call=True
)
def fetch_wind_vectors(n_clicks):
    """Fetch wind vectors from backend."""
    if n_clicks is None or n_clicks == 0:
        return None
    
    try:
        response = requests.get(
            "http://localhost:5000/api/wind-vectors",
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Wind vectors fetch failed: {e}")
        return None


@callback(
    Output('ocean-currents-store', 'data'),
    Input('toggle-currents-btn', 'n_clicks'),
    prevent_initial_call=True
)
def fetch_ocean_currents(n_clicks):
    """Fetch ocean currents from backend."""
    if n_clicks is None or n_clicks == 0:
        return None
    
    try:
        response = requests.get(
            "http://localhost:5000/api/ocean-currents",
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Ocean currents fetch failed: {e}")
        return None




@callback(
    [Output('toggle-wind-btn', 'style'),
     Output('toggle-currents-btn', 'style')],
    [Input('wind-vectors-store', 'data'),
     Input('ocean-currents-store', 'data'),
     Input('theme-store', 'data')]
)
def update_overlay_button_styles(wind_data, currents_data, theme):
    """Update button styles to show active state and respect theme."""
    colors = get_color_scheme(theme)
    
    # Wind button (blue base)
    wind_style = {
        'background': '#1e3a8a', 'border': '2px solid #3b82f6', 'color': '#ffffff',
        'fontSize': '12px', 'fontWeight': '600', 'padding': '8px 14px', 'borderRadius': '8px',
        'cursor': 'pointer', 'transition': 'all 0.3s ease', 'marginRight': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
    }
    if wind_data:
        wind_style['background'] = '#3b82f6'
        wind_style['border'] = '2px solid #1e40af'
        wind_style['boxShadow'] = '0 0 12px rgba(59, 130, 246, 0.5)'
    
    # Currents button (purple base)
    currents_style = {
        'background': '#5b21b6', 'border': '2px solid #a855f7', 'color': '#ffffff',
        'fontSize': '12px', 'fontWeight': '600', 'padding': '8px 14px', 'borderRadius': '8px',
        'cursor': 'pointer', 'transition': 'all 0.3s ease', 'marginRight': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
    }
    if currents_data:
        currents_style['background'] = '#a855f7'
        currents_style['border'] = '2px solid #7e22ce'
        currents_style['boxShadow'] = '0 0 12px rgba(168, 85, 247, 0.5)'
    
    return wind_style, currents_style




# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8000)
