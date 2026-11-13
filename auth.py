
import re
import streamlit as st

from database import (
    initialize_database,
    register_user,
    authenticate_user
)


def valid_email(email):
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return re.fullmatch(pattern, email.strip()) is not None


def valid_password(password):
    return (
        len(password) >= 8
        and len(password.encode("utf-8")) <= 72
        and any(c.isupper() for c in password)
        and any(c.islower() for c in password)
        and any(c.isdigit() for c in password)
    )


def show_authentication():
    initialize_database()

    st.markdown("""
    <style>
    .auth-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 30px;
    }

    .auth-subtitle {
        text-align: center;
        color: #a5b4fc;
        font-size: 16px;
        margin-bottom: 30px;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(
            90deg, #6366f1, #9333ea
        );
        color: white;
        border: none;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="auth-title">🧠 SentimentAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="auth-subtitle">'
        'Intelligent Sentiment Analysis Platform'
        '</div>',
        unsafe_allow_html=True
    )

    left, center, right = st.columns([1, 1.5, 1])

    with center:
        login_tab, register_tab = st.tabs([
            "🔐 Login",
            "📝 Register"
        ])

        with login_tab:
            st.subheader("Welcome Back")
            st.caption("Sign in to access your dashboard.")

            with st.form("login_form"):
                email = st.text_input(
                    "Email Address",
                    placeholder="Enter your email",
                    key="login_email"
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password"
                )

                submitted = st.form_submit_button(
                    "Login",
                    type="primary",
                    use_container_width=True
                )

            if submitted:
                if not email or not password:
                    st.error("Please fill in all fields.")
                else:
                    user = authenticate_user(
                        email, password
                    )

                    if user:
                        st.session_state["user"] = user
                        st.session_state["authenticated"] = True
                        st.rerun()
                    else:
                        st.error(
                            "Invalid email or password."
                        )

        with register_tab:
            st.subheader("Create Account")
            st.caption("Register to start analyzing reviews.")

            with st.form("register_form"):
                name = st.text_input(
                    "Full Name",
                    key="register_name"
                )

                email = st.text_input(
                    "Email Address",
                    key="register_email"
                )

                password = st.text_input(
                    "Create Password",
                    type="password",
                    key="register_password"
                )

                confirm = st.text_input(
                    "Confirm Password",
                    type="password",
                    key="confirm_password"
                )

                registered = st.form_submit_button(
                    "Create Account",
                    type="primary",
                    use_container_width=True
                )

            if registered:
                if not name.strip() or not email.strip():
                    st.error("Name and email are required.")

                elif not valid_email(email):
                    st.error("Enter a valid email address.")

                elif not valid_password(password):
                    st.error(
                        "Password must contain at least "
                        "8 characters, including uppercase, "
                        "lowercase and a number."
                    )

                elif password != confirm:
                    st.error("Passwords do not match.")

                else:
                    success, message = register_user(
                        name, email, password
                    )

                    if success:
                        st.success(
                            message + " Please log in."
                        )
                    else:
                        st.error(message)


def logout():
    st.session_state.clear()
    st.rerun()
