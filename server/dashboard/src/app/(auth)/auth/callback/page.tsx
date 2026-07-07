import { Suspense } from "react";
import OidcCallbackInner from "./callback-inner";

function CallbackSpinner() {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="size-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
    </div>
  );
}

// Server-component wrapper that provides the Suspense boundary required by
// useSearchParams() in OidcCallbackInner (Next.js 15 build requirement,
// matching the login page's pattern).
export default function AuthCallbackPage() {
  return (
    <Suspense fallback={<CallbackSpinner />}>
      <OidcCallbackInner />
    </Suspense>
  );
}
