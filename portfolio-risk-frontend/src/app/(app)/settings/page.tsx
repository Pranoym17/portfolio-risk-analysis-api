"use client"

import { useState } from "react"
import { 
  User,
  Bell,
  Shield,
  CreditCard,
  Palette,
  Link2,
  HelpCircle,
  ChevronRight,
  Check,
  Moon,
  Sun,
  Monitor,
  Mail,
  Smartphone,
  Globe,
  Camera,
  Trash2
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"

const settingsNav = [
  { id: "profile", label: "Profile", icon: User },
  { id: "notifications", label: "Notifications", icon: Bell },
  { id: "security", label: "Security", icon: Shield },
  { id: "billing", label: "Billing", icon: CreditCard },
  { id: "appearance", label: "Appearance", icon: Palette },
  { id: "integrations", label: "Integrations", icon: Link2 },
  { id: "help", label: "Help & Support", icon: HelpCircle },
]

const plans = [
  { id: "free", name: "Free", price: "$0", current: false },
  { id: "pro", name: "Pro", price: "$29/mo", current: true },
  { id: "enterprise", name: "Enterprise", price: "Custom", current: false },
]

const integrations = [
  { id: "coinbase", name: "Coinbase", description: "Sync your crypto holdings", connected: true, icon: "CB" },
  { id: "robinhood", name: "Robinhood", description: "Import stocks and ETFs", connected: false, icon: "RH" },
  { id: "fidelity", name: "Fidelity", description: "Reference integration placeholder", connected: true, icon: "FD" },
  { id: "schwab", name: "Charles Schwab", description: "Import investment accounts", connected: false, icon: "CS" },
]

export default function SettingsPage() {
  const [activeSection, setActiveSection] = useState("profile")
  const [theme, setTheme] = useState("dark")

  return (
    <div className="min-h-screen">
      {/* Page Header */}
      <div className="border-b border-border/60 bg-card/50">
        <div className="px-6 lg:px-8 py-6">
          <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Manage your account preferences and configurations
          </p>
        </div>
      </div>

      <div className="flex">
        {/* Settings Navigation */}
        <nav className="w-64 shrink-0 border-r border-border/60 min-h-[calc(100vh-8rem)]">
          <div className="p-4 space-y-1">
            {settingsNav.map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveSection(item.id)}
                className={cn(
                  "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors",
                  activeSection === item.id
                    ? "bg-primary/10 text-primary font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-surface-2"
                )}
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </button>
            ))}
          </div>
        </nav>

        {/* Settings Content */}
        <div className="flex-1 p-8 max-w-3xl">
          {activeSection === "profile" && (
            <div className="space-y-8">
              <div>
                <h2 className="text-lg font-semibold mb-1">Profile</h2>
                <p className="text-sm text-muted-foreground">
                  Manage your personal information and preferences
                </p>
              </div>

              {/* Avatar */}
              <div className="flex items-center gap-6">
                <div className="relative">
                  <div className="h-20 w-20 rounded-full bg-surface-2 flex items-center justify-center text-2xl font-semibold">
                    JD
                  </div>
                  <button className="absolute -bottom-1 -right-1 h-8 w-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center shadow-lg">
                    <Camera className="h-4 w-4" />
                  </button>
                </div>
                <div>
                  <p className="font-medium">Profile Photo</p>
                  <p className="text-sm text-muted-foreground">
                    JPG, PNG or GIF. Max 2MB.
                  </p>
                </div>
              </div>

              <Separator />

              {/* Form Fields */}
              <div className="grid gap-6">
                <div className="grid gap-2">
                  <Label htmlFor="name">Full Name</Label>
                  <Input id="name" defaultValue="John Doe" className="max-w-md" />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="email">Email Address</Label>
                  <Input id="email" type="email" defaultValue="john@example.com" className="max-w-md" />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="timezone">Timezone</Label>
                  <Select defaultValue="america-new-york">
                    <SelectTrigger className="max-w-md">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="america-new-york">Eastern Time (ET)</SelectItem>
                      <SelectItem value="america-chicago">Central Time (CT)</SelectItem>
                      <SelectItem value="america-denver">Mountain Time (MT)</SelectItem>
                      <SelectItem value="america-los-angeles">Pacific Time (PT)</SelectItem>
                      <SelectItem value="europe-london">GMT (London)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="currency">Display Currency</Label>
                  <Select defaultValue="usd">
                    <SelectTrigger className="max-w-md">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="usd">USD ($)</SelectItem>
                      <SelectItem value="eur">EUR (€)</SelectItem>
                      <SelectItem value="gbp">GBP (£)</SelectItem>
                      <SelectItem value="jpy">JPY (¥)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="flex justify-end">
                <Button>Save Changes</Button>
              </div>
            </div>
          )}

          {activeSection === "notifications" && (
            <div className="space-y-8">
              <div>
                <h2 className="text-lg font-semibold mb-1">Notifications</h2>
                <p className="text-sm text-muted-foreground">
                  Configure how and when you receive notifications
                </p>
              </div>

              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-lg bg-surface-2 flex items-center justify-center">
                      <Mail className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <div>
                      <p className="font-medium">Email Notifications</p>
                      <p className="text-sm text-muted-foreground">Receive updates via email</p>
                    </div>
                  </div>
                  <Switch defaultChecked />
                </div>

                <Separator />

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-lg bg-surface-2 flex items-center justify-center">
                      <Smartphone className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <div>
                      <p className="font-medium">Push Notifications</p>
                      <p className="text-sm text-muted-foreground">Get alerts on your mobile device</p>
                    </div>
                  </div>
                  <Switch defaultChecked />
                </div>

                <Separator />

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-lg bg-surface-2 flex items-center justify-center">
                      <Globe className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <div>
                      <p className="font-medium">Browser Notifications</p>
                      <p className="text-sm text-muted-foreground">Show desktop notifications</p>
                    </div>
                  </div>
                  <Switch />
                </div>
              </div>

              <Separator />

              <div>
                <h3 className="font-medium mb-4">Notification Types</h3>
                <div className="space-y-4">
                  {[
                    { label: "Price Alerts", description: "When assets hit target prices", defaultChecked: true },
                    { label: "Portfolio Updates", description: "Daily portfolio summaries", defaultChecked: true },
                    { label: "Market News", description: "Breaking financial news", defaultChecked: false },
                    { label: "Risk Warnings", description: "Unusual risk exposure alerts", defaultChecked: true },
                    { label: "Product Updates", description: "New features and improvements", defaultChecked: false },
                  ].map((item) => (
                    <div key={item.label} className="flex items-center justify-between">
                      <div>
                        <p className="text-sm font-medium">{item.label}</p>
                        <p className="text-xs text-muted-foreground">{item.description}</p>
                      </div>
                      <Switch defaultChecked={item.defaultChecked} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeSection === "appearance" && (
            <div className="space-y-8">
              <div>
                <h2 className="text-lg font-semibold mb-1">Appearance</h2>
                <p className="text-sm text-muted-foreground">
                  Customize the look and feel of the application
                </p>
              </div>

              {/* Theme Selection */}
              <div>
                <Label className="mb-3 block">Theme</Label>
                <div className="grid grid-cols-3 gap-4 max-w-md">
                  {[
                    { id: "light", label: "Light", icon: Sun },
                    { id: "dark", label: "Dark", icon: Moon },
                    { id: "system", label: "System", icon: Monitor },
                  ].map((option) => (
                    <button
                      key={option.id}
                      onClick={() => setTheme(option.id)}
                      className={cn(
                        "relative flex flex-col items-center gap-2 rounded-xl border-2 p-4 transition-colors",
                        theme === option.id
                          ? "border-primary bg-primary/5"
                          : "border-border hover:border-border/80 hover:bg-surface-1"
                      )}
                    >
                      <option.icon className="h-5 w-5" />
                      <span className="text-sm font-medium">{option.label}</span>
                      {theme === option.id && (
                        <div className="absolute top-2 right-2">
                          <Check className="h-4 w-4 text-primary" />
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              </div>

              <Separator />

              {/* Display Options */}
              <div className="space-y-4">
                <h3 className="font-medium">Display Options</h3>
                {[
                  { label: "Compact Mode", description: "Reduce spacing for more data density" },
                  { label: "Show Tooltips", description: "Display helpful hints on hover" },
                  { label: "Animate Charts", description: "Enable chart animations" },
                ].map((item) => (
                  <div key={item.label} className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">{item.label}</p>
                      <p className="text-xs text-muted-foreground">{item.description}</p>
                    </div>
                    <Switch defaultChecked />
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeSection === "billing" && (
            <div className="space-y-8">
              <div>
                <h2 className="text-lg font-semibold mb-1">Billing</h2>
                <p className="text-sm text-muted-foreground">
                  Manage your subscription and payment methods
                </p>
              </div>

              {/* Current Plan */}
              <div className="bg-surface-2/50 border border-border/60 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold">Pro Plan</h3>
                      <Badge className="bg-primary/20 text-primary hover:bg-primary/20">Current</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">$29/month, billed monthly</p>
                  </div>
                  <Button variant="outline">Change Plan</Button>
                </div>
                <div className="flex items-center gap-6 text-sm">
                  <div>
                    <span className="text-muted-foreground">Next billing date:</span>
                    <span className="ml-2 font-medium">May 15, 2026</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Payment method:</span>
                    <span className="ml-2 font-medium">•••• 4242</span>
                  </div>
                </div>
              </div>

              {/* Plan Comparison */}
              <div className="grid grid-cols-3 gap-4">
                {plans.map((plan) => (
                  <div
                    key={plan.id}
                    className={cn(
                      "rounded-xl border-2 p-5 transition-colors",
                      plan.current
                        ? "border-primary bg-primary/5"
                        : "border-border hover:border-border/80"
                    )}
                  >
                    <h4 className="font-semibold">{plan.name}</h4>
                    <p className="text-2xl font-bold mt-2">{plan.price}</p>
                    {plan.current && (
                      <Badge variant="secondary" className="mt-3">Current Plan</Badge>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeSection === "integrations" && (
            <div className="space-y-8">
              <div>
                <h2 className="text-lg font-semibold mb-1">Integrations</h2>
                <p className="text-sm text-muted-foreground">
                  Connect data sources and future integrations
                </p>
              </div>

              <div className="space-y-4">
                {integrations.map((integration) => (
                  <div
                    key={integration.id}
                    className="flex items-center justify-between p-4 bg-card border border-border/60 rounded-xl"
                  >
                    <div className="flex items-center gap-4">
                      <div className="h-12 w-12 rounded-lg bg-surface-2 flex items-center justify-center font-semibold text-sm">
                        {integration.icon}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <p className="font-medium">{integration.name}</p>
                          {integration.connected && (
                            <Badge variant="secondary" className="text-xs bg-positive/10 text-positive">
                              Connected
                            </Badge>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground">{integration.description}</p>
                      </div>
                    </div>
                    <Button variant={integration.connected ? "outline" : "default"} size="sm">
                      {integration.connected ? "Manage" : "Connect"}
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeSection === "security" && (
            <div className="space-y-8">
              <div>
                <h2 className="text-lg font-semibold mb-1">Security</h2>
                <p className="text-sm text-muted-foreground">
                  Manage your account security settings
                </p>
              </div>

              <div className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-card border border-border/60 rounded-xl">
                  <div>
                    <p className="font-medium">Two-Factor Authentication</p>
                    <p className="text-sm text-muted-foreground">Add an extra layer of security</p>
                  </div>
                  <Button variant="outline" size="sm">Enable</Button>
                </div>

                <div className="flex items-center justify-between p-4 bg-card border border-border/60 rounded-xl">
                  <div>
                    <p className="font-medium">Change Password</p>
                    <p className="text-sm text-muted-foreground">Update your account password</p>
                  </div>
                  <Button variant="outline" size="sm">Update</Button>
                </div>

                <div className="flex items-center justify-between p-4 bg-card border border-border/60 rounded-xl">
                  <div>
                    <p className="font-medium">Active Sessions</p>
                    <p className="text-sm text-muted-foreground">Manage your logged-in devices</p>
                  </div>
                  <Button variant="outline" size="sm">View All</Button>
                </div>
              </div>

              <Separator />

              <div className="p-4 bg-destructive/5 border border-destructive/20 rounded-xl">
                <div className="flex items-start gap-4">
                  <Trash2 className="h-5 w-5 text-destructive mt-0.5" />
                  <div className="flex-1">
                    <p className="font-medium text-destructive">Delete Account</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Permanently delete your account and all associated data. This action cannot be undone.
                    </p>
                    <Button variant="destructive" size="sm" className="mt-3">
                      Delete Account
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeSection === "help" && (
            <div className="space-y-8">
              <div>
                <h2 className="text-lg font-semibold mb-1">Help & Support</h2>
                <p className="text-sm text-muted-foreground">
                  Get help and find answers to your questions
                </p>
              </div>

              <div className="grid gap-4">
                {[
                  { label: "Documentation", description: "Learn how to use the portfolio risk workspace", href: "#" },
                  { label: "API Reference", description: "Integrate with our API", href: "#" },
                  { label: "Contact Support", description: "Get help from our team", href: "#" },
                  { label: "Community Forum", description: "Connect with other users", href: "#" },
                ].map((item) => (
                  <a
                    key={item.label}
                    href={item.href}
                    className="flex items-center justify-between p-4 bg-card border border-border/60 rounded-xl hover:bg-surface-1 transition-colors"
                  >
                    <div>
                      <p className="font-medium">{item.label}</p>
                      <p className="text-sm text-muted-foreground">{item.description}</p>
                    </div>
                    <ChevronRight className="h-5 w-5 text-muted-foreground" />
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
