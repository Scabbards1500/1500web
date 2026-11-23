"use client";

import { useEffect } from "react";
import { usePathname } from "next/navigation";
import { mountBlogDecor } from "@/lib/blog-decor";

// The animation owns this subtree and its photo/character layers.
const profileMarkup = `
  <div class="sidebar-top">
    <div class="avatar"><img src="/theme/profile.jpg" alt="Hanling Wang" width="160" height="160" /></div>
    <div class="side-info">
      <h1 class="sidebar-name">Hanling Wang <span aria-hidden="true">🐧</span></h1>
      <p class="sidebar-meta"><a href="mailto:scaaabbards@gmail.com">scaaabbards@gmail.com</a></p>
      <p class="sidebar-affil">M.S. in CSE<br />University of California, San Diego</p>
    </div>
  </div>
  <nav class="social-nav" aria-label="Social links">
    <a href="https://github.com/Scabbards1500" target="_blank" rel="noopener noreferrer">Github</a>
    <a href="https://www.linkedin.com/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
    <a href="https://scholar.google.com/citations?user=l3DCtDkAAAAJ" target="_blank" rel="noopener noreferrer">Google Scholar</a>
    <a href="https://blog.csdn.net/Scabbards_?spm=1000.2115.3001.5343" target="_blank" rel="noopener noreferrer">Blog</a>
    <a href="https://x.com/__Scabbard" target="_blank" rel="noopener noreferrer">Twitter</a>
  </nav>`;

export function ProfileSidebar() {
  return <aside className="sidebar profile-sidebar" dangerouslySetInnerHTML={{ __html: profileMarkup }} />;
}

export default function BlogDecor() {
  const pathname = usePathname();
  useEffect(() => mountBlogDecor(), [pathname]);
  return null;
}
