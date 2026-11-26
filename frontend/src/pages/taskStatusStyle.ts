// status -> [text colour, bar colour, tinted badge]. Shared by the task board
// and the /updates run panels so one status looks the same everywhere.
export interface StatusStyle { text: string; bar: string; badge: string; dot: string; line: string }

const STATUS_STYLE: Record<string, StatusStyle> = {
  running: { text: 'text-primary', bar: 'bg-primary', badge: 'border-primary/40 bg-primary/10 text-primary', dot: 'bg-primary', line: 'border-primary' },
  pending: { text: 'text-muted-foreground', bar: 'bg-muted-foreground', badge: 'border-border bg-muted text-muted-foreground', dot: 'bg-muted-foreground', line: 'border-muted-foreground' },
  completed: { text: 'text-success', bar: 'bg-success', badge: 'border-success/40 bg-success/10 text-success', dot: 'bg-success', line: 'border-success' },
  failed: { text: 'text-destructive', bar: 'bg-destructive', badge: 'border-destructive/40 bg-destructive/10 text-destructive', dot: 'bg-destructive', line: 'border-destructive' },
  timeout: { text: 'text-destructive', bar: 'bg-destructive', badge: 'border-destructive/40 bg-destructive/10 text-destructive', dot: 'bg-destructive', line: 'border-destructive' },
  cancelled: { text: 'text-muted-foreground', bar: 'bg-muted-foreground/60', badge: 'border-border bg-muted text-muted-foreground', dot: 'bg-muted-foreground/60', line: 'border-muted-foreground' },
};

export const styleOf = (status: string): StatusStyle => STATUS_STYLE[status] ?? STATUS_STYLE.pending;
