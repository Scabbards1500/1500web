import Image from "next/image";
import Link from "next/link";
import type { ReactNode } from "react";
import { CATEGORIES } from "@/lib/posts";
import { SearchBox, SearchMain, SearchProvider } from "./Search";

const navItems = [
  { label: "Homepage", href: "/" },
  ...CATEGORIES.map((category) => ({ label: category.label, href: `/blog/${category.slug}` })),
];

const socialLinks = [
  { label: "Github", href: "https://github.com/" },
  { label: "LinkedIn", href: "https://www.linkedin.com/" },
  { label: "Google Scholar", href: "https://scholar.google.com/" },
  { label: "Blog", href: "https://blog.csdn.net/Scabbards_?spm=1000.2115.3001.5343" },
  { label: "Twitter", href: "https://x.com/__Scabbard" },
];

type SiteLayoutProps = {
  children: ReactNode;
  active: string;
  /** 替换左侧个人信息栏的内容（如文章页的目录） */
  sidebar?: ReactNode;
  /** 阅读页使用更宽的整体布局和侧栏 */
  wide?: boolean;
};

export default function SiteLayout({ children, active, sidebar, wide = false }: SiteLayoutProps) {
  const containerWidth = wide ? "max-w-[1600px] md:grid-cols-[340px_1fr]" : "max-w-7xl md:grid-cols-[280px_1fr]";

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <div className={`relative mx-auto grid min-h-screen divide-y md:divide-y-0 md:divide-x md:divide-slate-200 ${containerWidth}`}>
        {sidebar ? (
          <aside className="bg-white shadow-md md:sticky md:top-0 md:h-screen md:max-h-screen">{sidebar}</aside>
        ) : (
          <ProfileSidebar />
        )}

        <SearchProvider>
          <div className="flex min-w-0 flex-col">
            <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur">
              {/* lg 以上不换行：搜索框展开时整组居中内容向左让开，而不是把搜索框挤到第二行 */}
              <nav className="mx-auto flex max-w-5xl flex-wrap items-center justify-center gap-x-3 gap-y-2 px-8 py-6 text-sm font-semibold uppercase tracking-wide text-slate-500 lg:flex-nowrap">
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
                <SearchBox />
              </nav>
            </header>

            <main className="flex-1 bg-slate-50 pb-16">
              <SearchMain>{children}</SearchMain>
            </main>
          </div>
        </SearchProvider>
      </div>
    </div>
  );
}

function ProfileSidebar() {
  return (
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
          <h1 className="text-xl font-semibold text-slate-900">
            Hanling Wang <span className="font-normal" aria-hidden>🦭</span>
          </h1>
          <p className="mt-2 text-sm text-slate-500">
            scaaabbards@gmail.com
          </p>
          <p className="mt-1 max-w-[16rem] text-sm leading-relaxed text-slate-500">
            M.S. in CSE
            <br />
            University of California, San Diego
          </p>
        </div>
      </div>

      <nav className="mt-12 flex flex-col gap-3 text-sm font-medium">
        {socialLinks.map((item) => (
          <Link
            key={item.label}
            href={item.href}
            className="transition-colors text-slate-500 hover:text-slate-900"
          >
            {item.label}
          </Link>
        ))}
      </nav>
    </aside>
  );
}
