import { AuthProvider } from "@/lib/auth";

export default function LoginLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <AuthProvider>{children}</AuthProvider>;
}
