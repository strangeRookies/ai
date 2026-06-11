import { Eye, EyeOff, KeyRound } from 'lucide-react';

interface AddCameraModalProps {
  newCamId: string;
  newCamLocation: string;
  newCamName: string;
  newCamPassword: string;
  showNewCamPw: boolean;
  onClose: () => void;
  onIdChange: (value: string) => void;
  onLocationChange: (value: string) => void;
  onNameChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onSubmit: () => void;
  onTogglePassword: () => void;
}

export function AddCameraModal(props: AddCameraModalProps) {
  const {
    newCamId,
    newCamLocation,
    newCamName,
    newCamPassword,
    showNewCamPw,
    onClose,
    onIdChange,
    onLocationChange,
    onNameChange,
    onPasswordChange,
    onSubmit,
    onTogglePassword,
  } = props;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="w-full max-w-md bg-[#071329] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
        <div className="px-5 py-4 bg-[#061224] border-b border-slate-800 flex items-center justify-between">
          <h3 className="text-sm font-extrabold text-white">Add camera</h3>
          <button onClick={onClose} className="text-xs text-slate-400 hover:text-white cursor-pointer">Close</button>
        </div>
        <div className="p-5 space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-400">Camera ID</label>
            <input value={newCamId} onChange={(event) => onIdChange(event.target.value)} className="w-full px-3 py-2.5 bg-[#020817] border border-slate-800 focus:border-blue-500 rounded-xl text-xs text-white outline-none" />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-400">Camera name</label>
            <input value={newCamName} onChange={(event) => onNameChange(event.target.value)} className="w-full px-3 py-2.5 bg-[#020817] border border-slate-800 focus:border-blue-500 rounded-xl text-xs text-white outline-none" />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-400">Location</label>
            <input value={newCamLocation} onChange={(event) => onLocationChange(event.target.value)} className="w-full px-3 py-2.5 bg-[#020817] border border-slate-800 focus:border-blue-500 rounded-xl text-xs text-white outline-none" />
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-400 flex items-center gap-1.5">
              <KeyRound className="w-3 h-3" />
              Camera password
            </label>
            <div className="relative">
              <input type={showNewCamPw ? 'text' : 'password'} value={newCamPassword} onChange={(event) => onPasswordChange(event.target.value)} className="w-full px-3 py-2.5 pr-10 bg-[#020817] border border-slate-800 focus:border-blue-500 rounded-xl text-xs text-white outline-none" />
              <button onClick={onTogglePassword} className="absolute right-3 top-2.5 text-slate-500 hover:text-slate-300 cursor-pointer">
                {showNewCamPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>
          <button onClick={onSubmit} className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl text-xs cursor-pointer">Register</button>
        </div>
      </div>
    </div>
  );
}
