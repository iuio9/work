package com.qczy.service;

/**
 * @author ：gwj
 * @date ：Created in 2024-10-24 18:53
 * @description：
 * @modified By：
 * @version: $
 */
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Service
public class DirectoryWatchService {

    private final ExecutorService executorService = Executors.newSingleThreadExecutor();

    public void watchDirectory(Path path) {
        executorService.submit(() -> {
            try (WatchService watchService = FileSystems.getDefault().newWatchService()) {
                // 注册目录监听事件
                path.register(watchService, StandardWatchEventKinds.ENTRY_CREATE,
                        StandardWatchEventKinds.ENTRY_DELETE,
                        StandardWatchEventKinds.ENTRY_MODIFY);

                WatchKey key;
                while ((key = watchService.take()) != null) {
                    for (WatchEvent<?> event : key.pollEvents()) {
                        // 获取事件类型和文件名
                        WatchEvent.Kind<?> kind = event.kind();
                        Path fileName = (Path) event.context();

                        System.out.println("Event kind: " + kind + ". File affected: " + fileName + ".");
                    }
                    // 重置 key 以继续监听
                    key.reset();
                }
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
        });
    }
}
