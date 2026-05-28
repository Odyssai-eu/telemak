import Foundation
import Hummingbird

struct ActivityHandler: Sendable {
    let activity: ActivityTracker

    func add(to router: Router<BasicRequestContext>) {
        router.get("/admin/activity") { _, _ async throws -> Response in
            try await self.handle()
        }
    }

    private func handle() async throws -> Response {
        let payload = await activity.snapshot()
        let data = try JSONEncoder().encode(payload)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }
}
