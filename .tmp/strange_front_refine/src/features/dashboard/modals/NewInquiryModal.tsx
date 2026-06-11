import { MessageSquare, Send } from 'lucide-react';
import type { InquiryCategory } from '../types/dashboard';
import { CATEGORIES, CATEGORY_ACTIVE_STYLES } from '../utils/dashboardStatus';

interface NewInquiryModalProps {
  qnaCategory: InquiryCategory;
  qnaContent: string;
  qnaTitle: string;
  onCategoryChange: (value: InquiryCategory) => void;
  onClose: () => void;
  onContentChange: (value: string) => void;
  onSubmit: () => void;
  onTitleChange: (value: string) => void;
}

export function NewInquiryModal({
  qnaCategory,
  qnaContent,
  qnaTitle,
  onCategoryChange,
  onClose,
  onContentChange,
  onSubmit,
  onTitleChange,
}: NewInquiryModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="w-full max-w-lg bg-[#071329] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
        <div className="px-5 py-4 bg-[#061224] border-b border-slate-800 flex items-center justify-between">
          <h3 className="text-sm font-extrabold text-white flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-blue-400" />
            New inquiry
          </h3>
          <button onClick={onClose} className="text-xs text-slate-400 hover:text-white cursor-pointer">Close</button>
        </div>
        <div className="p-5 space-y-4">
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-400">Category</label>
            <div className="grid grid-cols-2 gap-2">
              {CATEGORIES.map((category) => (
                <button
                  key={category}
                  onClick={() => onCategoryChange(category)}
                  className={`py-2.5 rounded-xl text-xs font-bold border cursor-pointer ${
                    qnaCategory === category
                      ? CATEGORY_ACTIVE_STYLES[category]
                      : 'bg-[#020817] border-slate-800 text-slate-400 hover:border-slate-600'
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-400">Title</label>
            <input value={qnaTitle} onChange={(event) => onTitleChange(event.target.value)} className="w-full px-3 py-2.5 bg-[#020817] border border-slate-800 focus:border-blue-500 rounded-xl text-xs text-white outline-none" />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-400">Content</label>
            <textarea value={qnaContent} onChange={(event) => onContentChange(event.target.value)} rows={5} className="w-full px-3 py-2.5 bg-[#020817] border border-slate-800 focus:border-blue-500 rounded-xl text-xs text-white outline-none resize-none" />
          </div>
          <button onClick={onSubmit} disabled={!qnaTitle.trim() || !qnaContent.trim()} className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-bold rounded-xl text-xs cursor-pointer flex items-center justify-center gap-2">
            <Send className="w-3.5 h-3.5" /> Submit inquiry
          </button>
        </div>
      </div>
    </div>
  );
}
