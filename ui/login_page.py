"""
Login gate for CreditIQ — session-based demo authentication (UI shell).
"""

from __future__ import annotations

import html

import streamlit as st

AUTH_FLAG = "auth_authenticated"
AUTH_USER = "auth_username"

# Demo credentials for placement / local demo (not production security)
_DEMO_USERS: dict[str, str] = {
    "analyst": "CreditIQ2024",
    "admin": "admin123",
    "risk_officer": "risk2024",
}


def is_authenticated() -> bool:
    return bool(st.session_state.get(AUTH_FLAG))


def display_name(username: str) -> str:
    name = username.strip()
    if "@" in name:
        return name.split("@")[0].replace(".", " ").title()
    return name.replace("_", " ").title()


def user_initial(username: str) -> str:
    shown = display_name(username)
    return (shown[0] if shown else "?").upper()


def authenticate(username: str, password: str) -> bool:
    user = username.strip().lower()
    if not user or not password:
        return False
    expected = _DEMO_USERS.get(user)
    if expected is None or password != expected:
        return False
    st.session_state[AUTH_FLAG] = True
    st.session_state[AUTH_USER] = user
    st.session_state.main_nav = "dashboard"
    return True


def logout() -> None:
    for key in (AUTH_FLAG, AUTH_USER):
        st.session_state.pop(key, None)


def inject_login_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }

        /* Center the login column on the viewport (wide layout safe) */
        section.main > div.block-container {
            width: 100% !important;
            max-width: 100% !important;
            padding: 7vh 1.25rem 2rem !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }

        section.main > div.block-container > [data-testid="stVerticalBlock"] {
            width: 100% !important;
            max-width: 700px !important;
            margin: 0 auto !important;
            align-items: stretch !important;
        }

        @media (max-width: 1024px) {
            section.main > div.block-container > [data-testid="stVerticalBlock"] {
                max-width: 650px !important;
            }
        }

        @media (max-width: 640px) {
            section.main > div.block-container {
                padding: 4vh 1rem 1.5rem !important;
            }
            section.main > div.block-container > [data-testid="stVerticalBlock"] {
                max-width: 100% !important;
            }
        }

        /* Glass authentication card */
        section.main [data-testid="stVerticalBlockBorderWrapper"] {
            width: 100% !important;
            max-width: 700px !important;
            margin: 0 auto !important;
            background: rgba(22, 29, 39, 0.92) !important;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(59, 130, 246, 0.14) !important;
            border-radius: 18px !important;
            padding: 2rem 2.25rem 1.65rem !important;
            box-shadow:
                0 28px 72px rgba(0, 0, 0, 0.48),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        @media (max-width: 1024px) {
            section.main [data-testid="stVerticalBlockBorderWrapper"] {
                max-width: 650px !important;
                padding: 1.75rem 1.85rem 1.5rem !important;
            }
        }

        @media (max-width: 640px) {
            section.main [data-testid="stVerticalBlockBorderWrapper"] {
                max-width: 100% !important;
                padding: 1.5rem 1.25rem 1.35rem !important;
                border-radius: 14px !important;
            }
        }

        /* Card header hierarchy */
        .login-card-header {
            text-align: center;
            margin-bottom: 1.65rem;
            padding-bottom: 0.25rem;
        }
        .login-card-logo {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.75rem;
            height: 2.75rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            font-size: 1.35rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.35);
        }
        .login-card-brand {
            font-size: 1.35rem;
            font-weight: 700;
            color: #f1f5f9;
            letter-spacing: -0.02em;
            margin: 0 0 1.1rem 0;
        }
        .login-card-brand .brand-accent { color: #3b82f6; }
        .login-card-title {
            font-size: 1.65rem;
            font-weight: 700;
            color: #f8fafc;
            letter-spacing: -0.03em;
            margin: 0 0 0.4rem 0;
            line-height: 1.2;
        }
        .login-card-sub {
            font-size: 0.88rem;
            color: #8b9bb4;
            margin: 0;
            line-height: 1.5;
        }

        section.main [data-testid="stForm"] {
            width: 100% !important;
            max-width: 100% !important;
        }

        section.main [data-testid="stForm"] [data-testid="stVerticalBlock"] {
            gap: 0.85rem !important;
            width: 100% !important;
        }

        section.main [data-testid="stForm"] label p {
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            color: #c5d0e0 !important;
        }

        section.main [data-testid="stForm"] input {
            width: 100% !important;
            max-width: 100% !important;
            background: rgba(15, 20, 28, 0.9) !important;
            border: 1px solid #2a3a52 !important;
            border-radius: 10px !important;
            color: #e8edf5 !important;
            min-height: 2.65rem;
            font-size: 0.9rem !important;
        }

        section.main [data-testid="stForm"] input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.35) !important;
        }

        /* Button: full width of card only; preserve Streamlit primary (red) styling */
        section.main [data-testid="stForm"] [data-testid="stFormSubmitButton"] {
            width: 100% !important;
            max-width: 100% !important;
        }
        section.main [data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
            width: 100% !important;
            max-width: 100% !important;
            border-radius: 10px !important;
            min-height: 2.65rem;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            margin-top: 0.35rem;
        }

        section.main [data-testid="stForm"] [data-testid="column"] {
            display: flex;
            align-items: center;
        }
        section.main [data-testid="stForm"] [data-testid="stCheckbox"] {
            margin: 0;
        }
        section.main [data-testid="stForm"] [data-testid="stCheckbox"] label {
            font-size: 0.78rem !important;
        }

        .login-forgot-wrap {
            width: 100%;
            text-align: right;
        }
        .login-forgot-wrap a {
            color: #3b82f6;
            font-size: 0.78rem;
            font-weight: 500;
            text-decoration: none;
        }
        .login-forgot-wrap a:hover {
            text-decoration: underline;
        }

        /* Alerts & demo expander stay within centered column */
        section.main [data-testid="stExpander"],
        section.main .stAlert {
            width: 100% !important;
            max-width: 700px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        @media (max-width: 1024px) {
            section.main [data-testid="stExpander"],
            section.main .stAlert {
                max-width: 650px !important;
            }
        }

        @media (max-width: 640px) {
            section.main [data-testid="stExpander"],
            section.main .stAlert {
                max-width: 100% !important;
            }
        }

        section.main [data-testid="stExpander"] {
            margin-top: 1rem !important;
        }
        section.main .stAlert {
            margin-top: 0.75rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_login_page() -> None:
    inject_login_styles()

    login_error: str | None = None

    with st.container(border=True):
        st.markdown(
            """
            <div class="login-card-header">
                <div class="login-card-logo">🏦</div>
                <p class="login-card-brand"><span class="brand-accent">Credit</span>IQ</p>
                <h1 class="login-card-title">Welcome back</h1>
                <p class="login-card-sub">Sign in to access your enterprise underwriting workspace.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="analyst")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            opt_left, opt_right = st.columns([1.15, 0.85], vertical_alignment="center")
            with opt_left:
                st.checkbox("Remember this session", value=True, disabled=True)
            with opt_right:
                st.markdown(
                    '<div class="login-forgot-wrap"><a href="#">Forgot password?</a></div>',
                    unsafe_allow_html=True,
                )

            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

    if submitted:
        if authenticate(username, password):
            st.rerun()
        login_error = "Invalid username or password. Please try again."

    if login_error:
        st.error(login_error)

    with st.expander("Demo accounts", expanded=False):
        st.markdown(
            """
            | Role | Username | Password |
            |------|----------|----------|
            | Analyst | `analyst` | `CreditIQ2024` |
            | Admin | `admin` | `admin123` |
            | Risk officer | `risk_officer` | `risk2024` |
            """
        )


def render_sidebar_user_block() -> None:
    """Bottom-left sidebar user card with avatar and sign out."""
    username = str(st.session_state.get(AUTH_USER, "user"))
    shown = html.escape(display_name(username))
    user_id = html.escape(username)
    initial = html.escape(user_initial(username))

    st.markdown(
        f"""
        <div class="sidebar-user-footer">
            <div class="sidebar-user-row">
                <div class="sidebar-user-avatar" aria-hidden="true">{initial}</div>
                <div class="sidebar-user-meta">
                    <span class="sidebar-user-name">{shown}</span>
                    <span class="sidebar-user-id">@{user_id}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Sign out", key="auth_sign_out", use_container_width=True, type="secondary"):
        logout()
        st.rerun()