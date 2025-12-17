"use client";

import { useState } from "react";
import PostList from "@/components/PostList";
import KeywordFilter from "@/components/KeywordFilter";
import type { Post } from "@/lib/posts";

type PublicationContentProps = {
  posts: Post[];
};

export default function PublicationContent({ posts }: PublicationContentProps) {
  const [filteredPosts, setFilteredPosts] = useState<Post[]>(posts);

  return (
    <>
      <KeywordFilter posts={posts} onFilterChange={setFilteredPosts} />

      <section className="mx-auto max-w-4xl px-6 pt-12 md:px-12 min-h-[calc(100vh-300px)]">
        <PostList posts={filteredPosts} />
      </section>
    </>
  );
}

