"use client"

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react"
import { useRouter } from "next/navigation"
import {
  TOKEN_STORAGE_KEY,
  getMe,
  login as loginRequest,
  signup as signupRequest,
} from "@/lib/api"
import type { User } from "@/lib/types"

type AuthContextValue = {
  user: User | null
  isLoading: boolean
  isAuthenticated: boolean
  refreshUser: () => Promise<User | null>
  login: (email: string, password: string) => Promise<User>
  signup: (email: string, password: string) => Promise<User>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const refreshUser = useCallback(async () => {
    if (typeof window === "undefined") return null

    const token = window.localStorage.getItem(TOKEN_STORAGE_KEY)
    if (!token) {
      setUser(null)
      setIsLoading(false)
      return null
    }

    try {
      const currentUser = await getMe()
      setUser(currentUser)
      return currentUser
    } catch {
      window.localStorage.removeItem(TOKEN_STORAGE_KEY)
      setUser(null)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    void refreshUser()
  }, [refreshUser])

  const handleAuthSuccess = useCallback((accessToken: string, nextUser: User) => {
    window.localStorage.setItem(TOKEN_STORAGE_KEY, accessToken)
    setUser(nextUser)
    return nextUser
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    const response = await loginRequest(email, password)
    return handleAuthSuccess(response.access_token, response.user)
  }, [handleAuthSuccess])

  const signup = useCallback(async (email: string, password: string) => {
    const response = await signupRequest(email, password)
    return handleAuthSuccess(response.access_token, response.user)
  }, [handleAuthSuccess])

  const logout = useCallback(() => {
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(TOKEN_STORAGE_KEY)
    }
    setUser(null)
    router.push("/login")
    router.refresh()
  }, [router])

  const value = useMemo<AuthContextValue>(() => ({
    user,
    isLoading,
    isAuthenticated: Boolean(user),
    refreshUser,
    login,
    signup,
    logout,
  }), [user, isLoading, refreshUser, login, signup, logout])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
