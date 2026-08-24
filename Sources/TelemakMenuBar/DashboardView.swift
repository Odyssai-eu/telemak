import SwiftUI

struct DashboardView: View {
    @ObservedObject var poller: HealthPoller
    @ObservedObject var settings: Settings

    private enum Section: String, CaseIterable, Identifiable {
        case activity = "Activity"
        case models = "Models"
        case chat = "Chat"
        case settings = "Settings"

        var id: String { rawValue }

        var icon: String {
            switch self {
            case .activity: return "gauge"
            case .models: return "shippingbox"
            case .chat: return "bubble.left.and.bubble.right"
            case .settings: return "gear"
            }
        }
    }

    @State private var selection: Section? = .activity

    var body: some View {
        NavigationSplitView {
            List(Section.allCases, selection: $selection) { section in
                Label(section.rawValue, systemImage: section.icon)
                    .tag(section)
            }
            .navigationSplitViewColumnWidth(min: 160, ideal: 180)
        } detail: {
            switch selection ?? .activity {
            case .activity:
                MonitorWindow(poller: poller, settings: settings)
            case .models:
                ModelsWindow(poller: poller, settings: settings)
            case .chat:
                ChatView(settings: settings)
            case .settings:
                SettingsView(settings: settings)
            }
        }
    }
}
