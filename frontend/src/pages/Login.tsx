import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useForm } from "react-hook-form";
import { supabase } from "../lib/supabase";
import styles from "./Login.module.css"

type AuthMode = "login" | "signup";

interface AuthFormData {
  email: string;
  password: string;
  username?: string;
}

export default function Login() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<AuthMode>("login");
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<AuthFormData>();

  const onSubmit = async (data: AuthFormData) => {
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      if (mode === "signup") {
        console.log("=== SIGNUP ATTEMPT ===");
        console.log("Email:", data.email);
        console.log("Name:", data.username);
        console.log("Supabase URL:", import.meta.env.VITE_SUPABASE_URL);

        const { data: signUpData, error: signUpError } =
          await supabase.auth.signUp({
            email: data.email,
            password: data.password,
            options: {
              data: {
                name: data.username,
              },
              emailRedirectTo: window.location.origin + '/',
            },
          });

        console.log("=== SIGNUP RESPONSE ===");
        console.log("Data:", signUpData);
        console.log("Error:", signUpError);
        console.log("User ID:", signUpData?.user?.id);
        console.log("Session:", signUpData?.session);

        if (signUpError) {
          console.error("Signup error details:", {
            message: signUpError.message,
            status: signUpError.status,
            name: signUpError.name,
          });
          throw signUpError;
        }

        setSuccess(
          "Account created successfully! Please check your email to verify your account."
        );
        reset();
        setLoading(false);
        setTimeout(() => setMode("login"), 3000);
      } else {
        console.log("=== LOGIN ATTEMPT ===");
        console.log("Email:", data.email);
        console.log("Supabase URL:", import.meta.env.VITE_SUPABASE_URL);
        console.log("Current URL:", window.location.href);

        const { data: signInData, error: signInError } =
          await supabase.auth.signInWithPassword({
            email: data.email,
            password: data.password,
            options: {
              // Ensure redirect stays on current origin
              emailRedirectTo: window.location.origin + '/',
            },
          });

        console.log("=== LOGIN RESPONSE ===");
        console.log("Data:", signInData);
        console.log("Error:", signInError);
        console.log("Session exists:", !!signInData?.session);
        console.log("User ID:", signInData?.user?.id);

        if (signInError) {
          console.error("Login error details:", {
            message: signInError.message,
            status: signInError.status,
            name: signInError.name,
          });
          throw signInError;
        }

        console.log("Login successful! Navigating to dashboard...");
        navigate("/");
      }
    } catch (err: any) {
      console.error("=== AUTH ERROR ===");
      console.error("Full error:", err);
      console.error("Error message:", err.message);
      console.error("Error status:", err.status);
      setError(err.message || "An error occurred during authentication");
      setLoading(false);
    }
  };

  const toggleMode = () => {
    setMode(mode === "login" ? "signup" : "login");
    setError(null);
    setSuccess(null);
    reset();
  };

  const handleGoogleSignIn = async () => {
    setGoogleLoading(true);
    setError(null);

    try {
      // Build redirect URL - use current origin for better localhost support
      const baseUrl = import.meta.env.BASE_URL || '/';
      // the dashboard lives at the root route
      const redirectTo = `${window.location.origin}${baseUrl}`.replace(/\/+/g, '/').replace(':/', '://');

      console.log('OAuth redirect URL:', redirectTo);

      const { error: signInError } = await supabase.auth.signInWithOAuth({
        provider: "google",
        options: {
          redirectTo,
        },
      });

      if (signInError) throw signInError;
    } catch (err: any) {
      console.error('Google sign in error:', err);
      setError(err.message || "An error occurred with Google sign in");
      setGoogleLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", flexDirection: "column", height: "100vh" }}>
      <div className={styles.background}></div>
      <div className={styles.backgroundMask}></div>
      <div className={styles.centerContainer}>
        <h1 className={styles.containerHeader}>{mode === "login" ? "muse-waffle" : "Create Account"}</h1>
        <form onSubmit={handleSubmit(onSubmit)} className={styles.inputGrid}>
          {mode === "signup" && (
            <div className="space-y-2">
              <label htmlFor="name" className={styles.inputLabel}>Name</label>
              <input
                id="name"
                placeholder="John Doe"
                className={styles.signInInput}
                {...register("username", {
                  required: mode === "signup" ? "Name is required" : false,
                })}
                disabled={loading}
              />
              {errors.username && (
                <p className="text-sm text-destructive">
                  {errors.username.message}
                </p>
              )}
            </div>
          )}

          <div className="space-y-2">
            <label htmlFor="email" className={styles.inputLabel}>
              Email
            </label>
            <input
              id="email"
              className={styles.signInInput}
              type="email"
              placeholder="you@example.com"
              {...register("email", {
                required: "Email is required",
                pattern: {
                  value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                  message: "Invalid email address",
                },
              })}
              disabled={loading}
            />
            {errors.email && (
              <p className="text-sm text-destructive">
                {errors.email.message}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <label htmlFor="password" className={styles.inputLabel}>
              Password
            </label>
            <input
              id="password"
              type="password"
              className={styles.signInInput}
              placeholder="••••••••"
              {...register("password", {
                required: "Password is required",
                minLength: {
                  value: 6,
                  message: "Password must be at least 6 characters",
                },
              })}
              disabled={loading}
            />
            {errors.password && (
              <p className="text-sm text-destructive">
                {errors.password.message}
              </p>
            )}
          </div>

          {error && (
            <p>
              {error}
            </p>
          )}

          {success && (
            <p>{success}</p>
          )}

          <button
            type="submit"
            className={styles.signInButton}
            disabled={loading || googleLoading}
          >
            {loading ? (
              <>
                {mode === "login" ? "Signing in..." : "Creating account..."}
              </>
            ) : mode === "login" ? (
              "Sign In"
            ) : (
              "Create Account"
            )}
          </button>

          <span className={styles.signInWithText}>
            OR
          </span>

          <button
            type="button"
            onClick={handleGoogleSignIn}
            className={styles.continueWithGoogle}
            disabled={loading || googleLoading}
          >
            {googleLoading ? (
              <>
                Connecting...
              </>
            ) : (
              <>
                <svg className={styles.googleSvg} viewBox="0 0 24 24">
                  <path
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                    fill="#4285F4"
                  />
                  <path
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                    fill="#34A853"
                  />
                  <path
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                    fill="#FBBC05"
                  />
                  <path
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                    fill="#EA4335"
                  />
                </svg>
                Continue with Google
              </>
            )}
          </button>
        </form>

            {/* <div className="mt-4 text-center text-sm">
              {mode === "login" ? (
                <>
                  Don't have an account?{" "}
                  <button
                    type="button"
                    onClick={toggleMode}
                    className="text-primary hover:underline font-medium"
                    disabled={loading}
                  >
                    Sign up
                  </button>
                </>
              ) : (
                <>
                  Already have an account?{" "}
                  <button
                    type="button"
                    onClick={toggleMode}
                    className="text-primary hover:underline font-medium"
                    disabled={loading}
                  >
                    Sign in
                  </button>
                </>
              )}
            </div> */}
      </div>
    </div>
  );
}
