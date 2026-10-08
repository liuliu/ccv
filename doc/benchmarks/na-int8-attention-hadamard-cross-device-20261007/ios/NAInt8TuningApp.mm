#import <UIKit/UIKit.h>
#include <cstdlib>
#include <string>
#include <vector>

int na_int8_tuning_run(int argc,char** argv);

@interface NAInt8TuningAppDelegate : UIResponder <UIApplicationDelegate>
@property (nonatomic, retain) UIWindow* window;
@end

@implementation NAInt8TuningAppDelegate

- (BOOL)application:(UIApplication*)application didFinishLaunchingWithOptions:(NSDictionary*)launchOptions
{
  application.idleTimerDisabled = YES;
  (void)launchOptions;

  self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
  UIViewController* const root = [UIViewController new];
  root.view.backgroundColor = UIColor.blackColor;
  self.window.rootViewController = root;
  [self.window makeKeyAndVisible];

  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    printf("thermal_state=%ld os=%s\n", (long)[NSProcessInfo processInfo].thermalState, [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);
    setenv("CCV_NA_WARMUP_SECONDS", "0.5", 1);
    setbuf(stdout, nullptr);
    setbuf(stderr, nullptr);
    NSArray<NSString*>* const args = [[NSProcessInfo processInfo] arguments];
    std::vector<std::string> storage;
    storage.reserve(args.count);
    storage.emplace_back("NAInt8TuningApp");
    for (NSUInteger i = 1; i < args.count; ++i) {
      NSString* const arg = args[i];
      storage.emplace_back(arg ? arg.UTF8String : "");
    }
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (std::string& arg : storage)
      argv.push_back(arg.data());
    const int status = na_int8_tuning_run((int)argv.size(), argv.data());
    printf("probe_exit=%d thermal_end=%ld\n",status,(long)[NSProcessInfo processInfo].thermalState);
    fflush(stdout);
    fflush(stderr);
    _Exit(status);
  });

  return YES;
}

@end

int main(int argc, char* argv[])
{
  @autoreleasepool {
    setbuf(stdout, nullptr); setbuf(stderr, nullptr);
    printf("thermal_state=%ld os=%s\n", (long)[NSProcessInfo processInfo].thermalState, [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);
    setenv("CCV_NA_WARMUP_SECONDS", "0.5", 1);
    const int result = na_int8_tuning_run(argc, argv);
    printf("probe_exit=%d thermal_end=%ld\n",result,(long)[NSProcessInfo processInfo].thermalState);
    fflush(stdout); fflush(stderr); return result;
  }
}
