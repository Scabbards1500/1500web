import Link from "next/link";
import type { ReactNode } from "react";
import { CATEGORIES } from "@/lib/posts";
import { SearchBox, SearchMain, SearchProvider } from "./Search";
import BlogDecor, { ProfileSidebar } from "./BlogDecor";

const navItems = [
  { label: "Homepage", href: "/" },
  ...CATEGORIES.map((category) => ({ label: category.label, href: `/blog/${category.slug}` })),
];

type SiteLayoutProps = {
  children: ReactNode;
  active: string;
  sidebar?: ReactNode;
  wide?: boolean;
};

export default function SiteLayout({ children, active, sidebar, wide = false }: SiteLayoutProps) {
  return (
    <div className={`site blog-site ${wide ? "reading-site" : ""}`}>
      <div className="site-shell">
        {sidebar ? <aside className="sidebar reading-sidebar">{sidebar}</aside> : <ProfileSidebar />}
        <SearchProvider>
          <div className="content-col">
            <header className="top-nav">
              <nav aria-label="Main navigation">
                {navItems.map((item) => (
                  <Link key={item.label} href={item.href} className={item.label === active ? "active" : ""} aria-current={item.label === active ? "page" : undefined}>
                    {item.label}
                  </Link>
                ))}
                <SearchBox />
              </nav>
            </header>
            <main className="main">
              <SearchMain>{children}</SearchMain>
            </main>
            <footer className="blog-footer"><span>SCABBARDS1500 / FIELD NOTES</span><span>Keep learning, keep exploring.</span></footer>
          </div>
        </SearchProvider>
      </div>
      <BlogDecor />
    </div>
  );
}
