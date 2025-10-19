/** @type {import('next').NextConfig} */
const repo = "1500web";
const isProd = process.env.NODE_ENV === "production";

export default {
  output: "export",               // 关键：导出静态站到 out/
  images: { unoptimized: true },  // 兼容 GH Pages
  basePath: isProd ? `/${repo}` : undefined,
  assetPrefix: isProd ? `/${repo}/` : undefined,
  trailingSlash: true,
};
