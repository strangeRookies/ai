import type { LucideIcon } from 'lucide-react';
import type { Inquiry } from '../../../shared/types/inquiry';

export interface IncidentAlert {
  id: string;
  time: string;
  timestamp: number;
  camera: string;
  type: string;
  label: string;
  severity: 'critical' | 'warning' | 'info';
  status: 'new' | 'resolved';
}

export interface RegisteredCamera {
  id: string;
  name: string;
  location: string;
  password?: string;
}

export interface MenuItemDefinition {
  id: MenuId;
  label: string;
  icon: LucideIcon;
  individualOnly: boolean;
}

export type MenuId =
  | 'home'
  | 'alerts'
  | 'history'
  | 'cameras'
  | 'mypage'
  | 'qna';

export type MypageTab = 'profile' | 'password' | 'notifications' | 'account';

export type InquiryCategory = Inquiry['category'];
