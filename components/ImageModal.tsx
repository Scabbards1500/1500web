"use client";

import { useEffect, useState, useRef } from "react";

type ImageModalProps = {
  src: string;
  alt: string;
  isOpen: boolean;
  onClose: () => void;
};

export default function ImageModal({ src, alt, isOpen, onClose }: ImageModalProps) {
  const [isAnimating, setIsAnimating] = useState(false);
  const [shouldRender, setShouldRender] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const scrollPositionRef = useRef<number>(0);

  // 防止背景滚动但保留滚动条
  useEffect(() => {
    if (isOpen) {
      // 保存当前滚动位置
      scrollPositionRef.current = window.scrollY;
      
      // 阻止滚动事件
      const preventScroll = (e: WheelEvent | TouchEvent) => {
        e.preventDefault();
      };
      
      // 阻止键盘滚动
      const preventKeyboardScroll = (e: KeyboardEvent) => {
        if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(e.key)) {
          e.preventDefault();
        }
      };

      // 添加事件监听器
      window.addEventListener('wheel', preventScroll, { passive: false });
      window.addEventListener('touchmove', preventScroll, { passive: false });
      window.addEventListener('keydown', preventKeyboardScroll);

      return () => {
        window.removeEventListener('wheel', preventScroll);
        window.removeEventListener('touchmove', preventScroll);
        window.removeEventListener('keydown', preventKeyboardScroll);
        // 恢复滚动位置
        window.scrollTo(0, scrollPositionRef.current);
      };
    }
  }, [isOpen]);

  useEffect(() => {
    // 清理之前的定时器
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    if (isOpen) {
      setShouldRender(true);
      // 等待 DOM 更新后触发动画
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          setIsAnimating(true);
        });
      });
    } else {
      // 开始关闭动画
      setIsAnimating(false);
      // 等待动画完成后再移除 DOM
      timeoutRef.current = setTimeout(() => {
        setShouldRender(false);
        timeoutRef.current = null;
      }, 300); // 与 transition 时间一致
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [isOpen]);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    if (isOpen) {
      window.addEventListener("keydown", handleEscape);
    }

    return () => {
      window.removeEventListener("keydown", handleEscape);
    };
  }, [isOpen, onClose]);

  if (!shouldRender) return null;

  return (
    <div
      className={`fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm transition-opacity duration-300 ease-out ${
        isAnimating ? "opacity-100" : "opacity-0"
      }`}
      onClick={onClose}
    >
      <div
        className={`relative max-h-[90vh] max-w-[90vw] p-4 transition-all duration-300 ease-out ${
          isAnimating ? "scale-100 opacity-100" : "scale-90 opacity-0"
        }`}
      >
        <img
          src={src}
          alt={alt}
          className="max-h-[90vh] max-w-[90vw] rounded-lg object-contain shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        />
        <button
          onClick={onClose}
          className="absolute right-2 top-2 rounded-full bg-black/50 p-2 text-white transition-all hover:bg-black/70 hover:scale-110"
          aria-label="关闭"
        >
          <svg
            className="h-6 w-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}

