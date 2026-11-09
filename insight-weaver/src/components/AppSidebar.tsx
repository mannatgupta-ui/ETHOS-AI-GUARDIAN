import { Upload, LayoutDashboard, BarChart3, Beaker, Wrench, Brain, Trophy, Sun, Moon, Filter, Scale, FlaskConical } from "lucide-react";
import { NavLink } from "@/components/NavLink";
import { useLocation } from "react-router-dom";
import { useData } from "@/context/DataContext";
import {
  Sidebar, SidebarContent, SidebarGroup, SidebarGroupContent, SidebarGroupLabel,
  SidebarMenu, SidebarMenuButton, SidebarMenuItem, SidebarFooter, useSidebar,
} from "@/components/ui/sidebar";

const workflow = [
  {
    group: "Preparation",
    items: [
      { title: "Upload", url: "/", icon: Upload },
      { title: "Overview", url: "/overview", icon: LayoutDashboard },
      { title: "Feature Selection", url: "/selection", icon: Filter },
    ]
  },
  {
    group: "Analysis",
    items: [
      { title: "EDA Dashboard", url: "/eda", icon: BarChart3 },
      { title: "Imputation Lab", url: "/imputation", icon: Beaker },
    ]
  },
  {
    group: "Modeling",
    items: [
      { title: "Modeling Prep", url: "/features", icon: Wrench },
      { title: "Fairness Lab", url: "/fairness", icon: Scale },
      { title: "PPI Debiasing", url: "/ppi", icon: FlaskConical },
      { title: "Model Lab", url: "/model", icon: Brain },
    ]
  },
  {
    group: "Export",
    items: [
      { title: "Results & Insights", url: "/results", icon: Trophy },
    ]
  }
];

export function AppSidebar() {
  const { state } = useSidebar();
  const collapsed = state === "collapsed";
  const location = useLocation();
  const { darkMode, setDarkMode } = useData();

  return (
    <Sidebar collapsible="icon" className="border-r border-sidebar-border">
      <SidebarContent className="pt-4">
        <SidebarGroup>
          {!collapsed && (
            <div className="px-4 pb-4 mb-2 border-b border-sidebar-border">
              <h1 className="font-display text-xl font-black tracking-tighter text-primary">Ethos AI <span className="italic">Guardian</span></h1>
              <p className="text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground/60 mt-0.5">Ethical Audit Console</p>
            </div>
          )}
          {workflow.map((section) => (
            <div key={section.group} className="mb-4">
              <SidebarGroupLabel className="text-[9px] font-black uppercase tracking-widest text-muted-foreground/40 mb-2 px-3">
                {section.group}
              </SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {section.items.map((item) => (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton asChild>
                        <NavLink
                          to={item.url}
                          end={item.url === "/"}
                          className="flex items-center gap-3 px-3 py-2 rounded-lg text-xs transition-all duration-200 text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                          activeClassName="bg-primary/10 text-primary font-bold shadow-glow border-l-2 border-primary"
                        >
                          <item.icon className="h-3.5 w-3.5 shrink-0" />
                          {!collapsed && <span className="tracking-tight">{item.title}</span>}
                        </NavLink>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </div>
          ))}
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="p-3 border-t border-sidebar-border">
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm text-sidebar-foreground hover:bg-sidebar-accent transition-colors"
        >
          {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          {!collapsed && <span>{darkMode ? "Light Mode" : "Dark Mode"}</span>}
        </button>
      </SidebarFooter>
    </Sidebar>
  );
}
