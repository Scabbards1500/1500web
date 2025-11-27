import Image from "next/image";
import Link from "next/link";
import type { ReactNode } from "react";

const navItems = [
  { label: "Homepage", href: "/" },
  { label: "Publication", href: "/publication" },
  { label: "Internship", href: "/internship" },
  { label: "Gallery", href: "/gallery" },
  { label: "Notes", href: "/notes" },
];

const socialLinks = [
  { label: "Scabbards", href: "#", emphasis: true },
  { label: "Github", href: "https://github.com/" },
  { label: "LinkedIn", href: "https://www.linkedin.com/" },
  { label: "Google Scholar", href: "https://scholar.google.com/" },
];

type SiteLayoutProps = {
  children: ReactNode;
  active: string;
};

export default function SiteLayout({ children, active }: SiteLayoutProps) {
  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <div className="relative mx-auto grid min-h-screen max-w-7xl divide-y md:grid-cols-[280px_1fr] md:divide-y-0 md:divide-x">
        <aside className="flex flex-col justify-between bg-white px-8 py-12 shadow-md md:sticky md:top-0 md:h-screen md:max-h-screen">
          <div className="flex flex-col items-center gap-6">
            <div className="flex h-40 w-40 items-center justify-center overflow-hidden rounded-full border-4 border-slate-200 bg-slate-100">
              <Image
                src="/profile.jpg"
                alt="Scabbards 个人照片"
                width={160}
                height={160}
                priority
                className="h-full w-full object-cover"
              />
            </div>
            <div className="text-center">
              <h1 className="text-xl font-semibold text-slate-900">Scabbards</h1>
              <p className="mt-2 text-sm text-slate-500">
                科研爱好者，关注传感、机器学习与城市韧性建设。
              </p>
            </div>
          </div>

          <nav className="mt-12 flex flex-col gap-3 text-sm font-medium">
            {socialLinks.map((item) => (
              <Link
                key={item.label}
                href={item.href}
                className={`transition-colors hover:text-slate-900 ${
                  item.emphasis ? "text-lg font-semibold text-slate-900" : "text-slate-500"
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </aside>

        <div className="flex flex-col">
          <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur">
            <nav className="mx-auto flex max-w-4xl items-center justify-center gap-6 px-8 py-6 text-sm font-semibold uppercase tracking-wide text-slate-500">
              {navItems.map((item) => (
                <Link
                  key={item.label}
                  href={item.href}
                  className={`rounded-md px-3 py-2 transition-all hover:bg-slate-900 hover:text-white ${
                    item.label === active ? "bg-slate-900 text-white" : ""
                  }`}
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </header>

          <main className="flex-1 bg-slate-50 pb-16">{children}</main>
        </div>
      </div>
    </div>
  );
}

