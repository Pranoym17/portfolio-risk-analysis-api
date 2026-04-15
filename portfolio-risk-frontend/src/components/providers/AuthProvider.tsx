"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { TOKEN_STORAGE_KEY, getMe, login, signup } from "@/lib/api";
import type { User } from "@/lib/types";
import { getErrorMessage } from "@/lib/utils";

type AuthContextValue = {
  user: User | null;
  loading: boolean;
  token: string | null;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string) => Promise<void>;
  signOut: () => void;
  refreshUser: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const existing = window.localStorage.getItem(TOKEN_STORAGE_KEY);
    if (!existing) {
      queueMicrotask(() => {
        setLoading(false);
      });
      return;
    }

    getMe()
      .then((nextUser) => {
        setUser(nextUser);
        setToken(existing);
      })
      .catch(() => {
        window.localStorage.removeItem(TOKEN_STORAGE_KEY);
        setToken(null);
        setUser(null);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  async function refreshUser() {
    const nextUser = await getMe();
    setUser(nextUser);
  }

  async function completeAuth(authPromise: Promise<{ access_token: string; user: User }>) {
    const payload = await authPromise;
    window.localStorage.setItem(TOKEN_STORAGE_KEY, payload.access_token);
    setToken(payload.access_token);
    setUser(payload.user);
    setLoading(false);
    router.push("/dashboard");
    router.refresh();
  }

  async function signIn(email: string, password: string) {
    try {
      await completeAuth(login(email, password));
      toast.success("Signed in");
    } catch (error: unknown) {
      toast.error(getErrorMessage(error, "Unable to sign in"));
      throw error;
    }
  }

  async function signUp(email: string, password: string) {
    try {
      await completeAuth(signup(email, password));
      toast.success("Account created");
    } catch (error: unknown) {
      toast.error(getErrorMessage(error, "Unable to create account"));
      throw error;
    }
  }

  function signOut() {
    window.localStorage.removeItem(TOKEN_STORAGE_KEY);
    setToken(null);
    setUser(null);
    router.push("/login");
    router.refresh();
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        token,
        signIn,
        signUp,
        signOut,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
