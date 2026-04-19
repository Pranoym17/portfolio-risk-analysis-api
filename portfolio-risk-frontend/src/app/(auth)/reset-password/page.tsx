"use client"

import { useState } from "react"
import Link from "next/link"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Eye, EyeOff, Loader2, Check, CheckCircle2 } from "lucide-react"

const passwordRequirements = [
  { label: "At least 8 characters", check: (p: string) => p.length >= 8 },
  { label: "One uppercase letter", check: (p: string) => /[A-Z]/.test(p) },
  { label: "One number", check: (p: string) => /[0-9]/.test(p) },
]

export default function ResetPasswordPage() {
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (password !== confirmPassword) return
    setIsLoading(true)
    await new Promise((resolve) => setTimeout(resolve, 1500))
    setIsLoading(false)
    setSubmitted(true)
  }

  const allRequirementsMet = passwordRequirements.every((req) => req.check(password))
  const passwordsMatch = password === confirmPassword && password.length > 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {submitted ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <div className="inline-flex items-center justify-center h-16 w-16 rounded-full bg-positive/10 text-positive mb-6">
            <CheckCircle2 className="h-8 w-8" />
          </div>
          <h1 className="text-2xl font-bold text-foreground">Password reset successful</h1>
          <p className="mt-3 text-muted-foreground">
            Your password has been successfully reset. You can now sign in with your new password.
          </p>
          <Button className="mt-8 h-11 px-8" asChild>
            <Link href="/login">Sign in</Link>
          </Button>
        </motion.div>
      ) : (
        <>
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-foreground">Set new password</h1>
            <p className="mt-2 text-muted-foreground">
              Please create a new password for your account.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="space-y-2">
              <Label htmlFor="password">New password</Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter new password"
                  className="h-11 bg-surface-1 pr-10"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
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

            <div className="space-y-2">
              <Label htmlFor="confirmPassword">Confirm password</Label>
              <div className="relative">
                <Input
                  id="confirmPassword"
                  type={showConfirm ? "text" : "password"}
                  placeholder="Confirm new password"
                  className="h-11 bg-surface-1 pr-10"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowConfirm(!showConfirm)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showConfirm ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
              {confirmPassword && !passwordsMatch && (
                <p className="text-xs text-destructive">Passwords do not match</p>
              )}
              {passwordsMatch && (
                <p className="text-xs text-positive flex items-center gap-1">
                  <Check className="h-3 w-3" />
                  Passwords match
                </p>
              )}
            </div>

            <Button
              type="submit"
              className="w-full h-11"
              disabled={isLoading || !allRequirementsMet || !passwordsMatch}
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Resetting...
                </>
              ) : (
                "Reset password"
              )}
            </Button>
          </form>
        </>
      )}
    </motion.div>
  )
}
