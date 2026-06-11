import {
  Bell,
  Calendar,
  Camera,
  HelpCircle,
  Tv,
  User,
} from 'lucide-react';
import type {
  InquiryCategory,
  IncidentAlert,
  MenuItemDefinition,
  RegisteredCamera,
} from '../types/dashboard';

export const INITIAL_ALERTS: IncidentAlert[] = [];

export const INITIAL_CAMERAS: RegisteredCamera[] = [
  { id: 'CCTV-01', name: 'Room 1', location: '1F', password: 'cam1234' },
  { id: 'CCTV-02', name: 'Corridor A', location: '1F', password: 'hall5678' },
  { id: 'CCTV-03', name: 'Room 2', location: '1F' },
];

export const MOCK_LOGIN_HISTORY = [
  { date: '2026-05-29 09:42', device: 'Chrome / Windows 11', ip: '192.168.1.xxx', status: 'Success' },
  { date: '2026-05-28 17:15', device: 'Chrome / Windows 11', ip: '192.168.1.xxx', status: 'Success' },
  { date: '2026-05-27 08:30', device: 'Safari / macOS', ip: '192.168.2.xxx', status: 'Success' },
  { date: '2026-05-26 13:22', device: 'Chrome / Android', ip: '10.0.0.xxx', status: 'Failed' },
] as const;

export const ALL_MENU_ITEMS: readonly MenuItemDefinition[] = [
  { id: 'home', label: 'Dashboard', icon: Tv, individualOnly: false },
  { id: 'alerts', label: 'Alerts', icon: Bell, individualOnly: false },
  { id: 'history', label: 'History', icon: Calendar, individualOnly: false },
  { id: 'cameras', label: 'Cameras', icon: Camera, individualOnly: true },
  { id: 'mypage', label: 'My Page', icon: User, individualOnly: false },
  { id: 'qna', label: 'Q&A', icon: HelpCircle, individualOnly: false },
];

export const CATEGORY_STYLES: Record<InquiryCategory, string> = {
  '카메라 및 영상': 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  '알림 및 경보': 'bg-rose-500/10 text-rose-400 border-rose-500/20',
  '모바일': 'bg-violet-500/10 text-violet-400 border-violet-500/20',
  '기타': 'bg-slate-500/10 text-slate-400 border-slate-500/20',
};

export const CATEGORY_ACTIVE_STYLES: Record<InquiryCategory, string> = {
  '카메라 및 영상': 'bg-blue-600/20 text-blue-300 border-blue-400/50',
  '알림 및 경보': 'bg-rose-600/20 text-rose-300 border-rose-400/50',
  '모바일': 'bg-violet-600/20 text-violet-300 border-violet-400/50',
  '기타': 'bg-slate-600/20 text-slate-300 border-slate-400/50',
};

export const CATEGORIES: InquiryCategory[] = ['카메라 및 영상', '알림 및 경보', '모바일', '기타'];

export const CCTV_FEEDS = [
  { id: 'CCTV-01', name: 'Room 1', style: 'brightness-90 contrast-100 hue-rotate-15' },
  { id: 'CCTV-02', name: 'Corridor A', style: 'border-2 border-rose-500 animate-pulse', alert: true },
  { id: 'CCTV-03', name: 'Room 2', style: 'brightness-105 contrast-95' },
  { id: 'CCTV-04', name: 'Waiting 1', style: 'saturate-50 contrast-110' },
  { id: 'CCTV-05', name: 'Waiting 2', style: 'hue-rotate-60' },
  { id: 'CCTV-06', name: 'Entrance', style: 'brightness-75' },
  { id: 'CCTV-07', name: 'Front Door', style: 'brightness-110 contrast-105' },
] as const;

export function eventButtonStyle(severity: 'critical' | 'warning' | 'info') {
  if (severity === 'critical') return 'bg-[#ef4444] hover:bg-red-400';
  if (severity === 'warning') return 'bg-[#f59e0b] hover:bg-amber-400';
  return 'bg-[#334155] hover:bg-slate-500';
}

export function getPasswordStrength(pw: string): { level: number; label: string; color: string } {
  if (!pw) return { level: 0, label: '', color: '' };
  const hasLower = /[a-z]/.test(pw);
  const hasUpper = /[A-Z]/.test(pw);
  const hasNum = /[0-9]/.test(pw);
  const hasSpec = /[^a-zA-Z0-9]/.test(pw);
  const score = [pw.length >= 8, hasLower, hasUpper, hasNum, hasSpec].filter(Boolean).length;
  if (score <= 2) return { level: 1, label: 'Weak', color: 'bg-red-500' };
  if (score <= 3) return { level: 2, label: 'Medium', color: 'bg-amber-500' };
  return { level: 3, label: 'Strong', color: 'bg-emerald-500' };
}
