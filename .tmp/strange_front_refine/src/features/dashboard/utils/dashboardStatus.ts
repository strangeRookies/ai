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
  { id: 'CCTV-01', name: '병실 1', location: '1층 병실 구역', password: 'cam1234' },
  { id: 'CCTV-02', name: '복도 A', location: '1층 복도', password: 'hall5678' },
  { id: 'CCTV-03', name: '병실 2', location: '1층 병실 구역' },
];

export const MOCK_LOGIN_HISTORY = [
  { date: '2026-05-29 09:42', device: 'Chrome / Windows 11', ip: '192.168.1.xxx', status: '성공' },
  { date: '2026-05-28 17:15', device: 'Chrome / Windows 11', ip: '192.168.1.xxx', status: '성공' },
  { date: '2026-05-27 08:30', device: 'Safari / macOS', ip: '192.168.2.xxx', status: '성공' },
  { date: '2026-05-26 13:22', device: 'Chrome / Android', ip: '10.0.0.xxx', status: '실패' },
] as const;

export const ALL_MENU_ITEMS: readonly MenuItemDefinition[] = [
  { id: 'home', label: '대시보드', icon: Tv, individualOnly: false },
  { id: 'alerts', label: '이벤트 알림', icon: Bell, individualOnly: false },
  { id: 'history', label: '이력 조회', icon: Calendar, individualOnly: false },
  { id: 'cameras', label: '카메라 관리', icon: Camera, individualOnly: true },
  { id: 'mypage', label: '내 정보', icon: User, individualOnly: false },
  { id: 'qna', label: '문의 내역', icon: HelpCircle, individualOnly: false },
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

export function getDefaultInquiryCategory() {
  return CATEGORIES[3];
}

export function getDefaultSearchCameraLabel() {
  return '전체';
}

export function getConnectionStateCopy(connectionState: 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error') {
  if (connectionState === 'connected') return '실시간 모니터링 중입니다.';
  if (connectionState === 'connecting') return '이벤트 정보를 불러오는 중입니다.';
  if (connectionState === 'disconnected' || connectionState === 'error') return '이벤트 채널 연결 상태를 확인해주세요.';
  return '모니터링을 준비하고 있습니다.';
}

export function getConnectionStateTone(connectionState: 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error') {
  if (connectionState === 'connected') return '#34d399';
  if (connectionState === 'connecting') return '#fbbf24';
  return '#94a3b8';
}

export function getConnectionStateDot(connectionState: 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error') {
  if (connectionState === 'connected') return '#34d399';
  if (connectionState === 'connecting') return '#fbbf24';
  return '#64748b';
}

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
  if (score <= 2) return { level: 1, label: '약함', color: 'bg-red-500' };
  if (score <= 3) return { level: 2, label: '보통', color: 'bg-amber-500' };
  return { level: 3, label: '강함', color: 'bg-emerald-500' };
}
