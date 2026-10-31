/** Proxy /api/* to the FastAPI backend so the browser never needs CORS. */
const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${BACKEND}/api/:path*` }];
  },
};
export default nextConfig;
