import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  try {
    const { path: pathArray } = await params;
    const filePath = path.join(process.cwd(), "content", ...pathArray);
    
    // 安全检查：确保路径在 content 目录内
    const contentDir = path.join(process.cwd(), "content");
    const resolvedPath = path.resolve(filePath);
    const resolvedContentDir = path.resolve(contentDir);
    
    if (!resolvedPath.startsWith(resolvedContentDir)) {
      return new NextResponse("Forbidden", { status: 403 });
    }

    if (!fs.existsSync(filePath)) {
      return new NextResponse("Not Found", { status: 404 });
    }

    const fileBuffer = fs.readFileSync(filePath);
    const ext = path.extname(filePath).toLowerCase();
    
    // 根据文件扩展名设置 Content-Type
    const contentTypeMap: Record<string, string> = {
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".gif": "image/gif",
      ".webp": "image/webp",
      ".svg": "image/svg+xml",
    };

    const contentType = contentTypeMap[ext] || "application/octet-stream";

    return new NextResponse(fileBuffer, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    });
  } catch (error) {
    console.error("Error serving content file:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}

