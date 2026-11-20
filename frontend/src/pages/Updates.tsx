import ScheduleCard from '../components/updates/ScheduleCard';
import { apiErrorText, useScheduleKinds, useSchedules } from '../api/hooks/useSchedules';

/** Scheduled jobs: when they run, what the last runs did, the current run live. */
export default function Updates() {
  const schedules = useSchedules();
  const kinds = useScheduleKinds();

  if (schedules.isError) {
    return <div className="py-16 text-center text-sm text-destructive">{apiErrorText(schedules.error)}</div>;
  }
  if (!schedules.data) {
    return <div className="py-16 text-center text-sm text-muted-foreground">加载中…</div>;
  }
  if (schedules.data.length === 0) {
    return <div className="py-16 text-center text-sm text-muted-foreground">暂无计划</div>;
  }
  return (
    <div className="grid gap-4">
      {schedules.data.map((schedule) => (
        <ScheduleCard key={schedule.id} schedule={schedule} kind={kinds.data?.find((k) => k.name === schedule.kind)} />
      ))}
    </div>
  );
}
