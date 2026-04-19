"use client"

import { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { motion } from "framer-motion"
import { Eye, EyeOff, Loader2, Check } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useAuth } from "@/components/providers/auth-provider"
import { getErrorMessage } from "@/lib/utils"

const passwordRequirements = [
  { label: "At least 8 characters", check: (p: string) => p.length >= 8 },
  { label: "One uppercase letter", check: (p: string) => /[A-Z]/.test(p) },
  { label: "One number", check: (p: string) => /[0-9]/.test(p) },
]

export default function SignupPage() {
  const router = useRouter()
  const { signup } = useAuth()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      await signup(email, password)
      toast.success("Account created")
      router.push("/dashboard")
      router.refresh()
    } catch (error) {
      toast.error(getErrorMessage(error, "Unable to create account"))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-foreground">Create your account</h1>
        <p className="mt-2 text-muted-foreground">
          Open a new analysis workspace and start building portfolios.
        </p>
      </div>

      <div className="mb-6 rounded-xl border border-border/60 bg-surface-1/70 p-4">
        <p className="text-sm font-medium text-foreground">Backend-backed signup</p>
        <p className="mt-1 text-sm text-muted-foreground">
          New accounts are created directly against your FastAPI auth endpoints and signed in automatically.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        <div className="space-y-2">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            placeholder="name@example.com"
            className="h-11 bg-surface-1"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            required
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="password">Password</Label>
          <div className="relative">
            <Input
              id="password"
              type={showPassword ? "text" : "password"}
              placeholder="Create a password"
              className="h-11 bg-surface-1 pr-10"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="new-password"
              required
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
            >
              {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              <span className="sr-only">
                {showPassword ? "Hide password" : "Show password"}
              </span>
            </button>
          </div>

          {password && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="pt-2 space-y-1.5"
            >
              {passwordRequirements.map((req) => {
                const met = req.check(password)
                return (
                  <div
                    key={req.label}
                    className={`flex items-center gap-2 text-xs ${
                      met ? "text-positive" : "text-muted-foreground"
                    }`}
                  >
                    <Check className={`h-3 w-3 ${met ? "opacity-100" : "opacity-30"}`} />
                    {req.label}
                  </div>
                )
              })}
            </motion.div>
          )}
        </div>

        <Button type="submit" className="w-full h-11" disabled={isLoading}>
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Creating account...
            </>
          ) : (
            "Create account"
          )}
        </Button>
      </form>

      <p className="mt-8 text-center text-sm text-muted-foreground">
        Already have an account?{" "}
        <Link href="/login" className="text-primary hover:underline font-medium">
          Sign in
        </Link>
      </p>
    </motion.div>
  )
}
