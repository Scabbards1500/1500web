import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const geistSans = localFont({ src: "../public/fonts/geist-latin.woff2", variable: "--font-geist-sans", display: "swap" });
const geistMono = localFont({ src: "../public/fonts/geist-mono-latin.woff2", variable: "--font-geist-mono", display: "swap" });

export const metadata: Metadata = {
  title: "Hanling Wang",
  description: "Research notes, engineering experiments, and course notes by Hanling Wang.",
  icons: {
    icon: "/tuxdo.svg",
    shortcut: "/tuxdo.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: `try{document.documentElement.dataset.theme=localStorage.getItem("site-theme")||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light")}catch(e){document.documentElement.dataset.theme="light"}` }} /></head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
