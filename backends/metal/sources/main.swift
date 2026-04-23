import Foundation
import AppKit

let defaults = loadDefaults()
let mode = parseArguments(CommandLine.arguments, defaults: defaults)

switch mode {
case .render(let options):
    renderVoronoiFromSites(options)
    exit(0)
case .train(let options):
    if options.showViewer {
        let app = NSApplication.shared
        app.setActivationPolicy(.regular)
        DispatchQueue.global(qos: .userInitiated).async {
            trainVoronoi(options)
            DispatchQueue.main.async {
                NSApp.terminate(nil)
            }
            print("\nTraining complete!")
        }
        app.run()
    } else {
        trainVoronoi(options)
        print("\nTraining complete!")
    }
}
