"use client";

import { useState } from "react";
import PostList from "@/components/PostList";
import YearFilter from "@/components/YearFilter";
import type { Post } from "@/lib/posts";

type HomeContentProps = {
  posts: Post[];
};

export default function HomeContent({ posts }: HomeContentProps) {
  const [filteredPosts, setFilteredPosts] = useState<Post[]>(posts);

  return (
    <>
      <YearFilter posts={posts} onFilterChange={setFilteredPosts} />

      <section className="mx-auto flex max-w-5xl flex-col gap-6 px-6 pt-16 md:px-12">
        <PostList posts={filteredPosts} />
      </section>
    </>
  );
}

