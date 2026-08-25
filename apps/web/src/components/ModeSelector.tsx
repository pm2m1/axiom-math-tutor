export type Mode = 'tutor' | 'check' | 'practice'

interface ModeOption {
  id: Mode
  label: string
}

interface ModeSelectorProps {
  modes: ModeOption[]
  active: Mode
  onSelect: (mode: Mode) => void
}

export function ModeSelector({ modes, active, onSelect }: ModeSelectorProps) {
  return (
    <div className="mode-selector" role="tablist" aria-label="Tutoring mode">
      {modes.map((mode) => (
        <button
          key={mode.id}
          type="button"
          role="tab"
          aria-selected={mode.id === active}
          className={mode.id === active ? 'mode-tab mode-tab-active' : 'mode-tab'}
          onClick={() => onSelect(mode.id)}
        >
          {mode.label}
        </button>
      ))}
    </div>
  )
}
