from collections import deque

class TemperatureLogger:
    def __init__(self):
        self.readings = deque()

    def addReading(self, temp: int):
        self.readings.append(temp)

    def getAverage(self, k: int) -> float:
        if k <= 0 or len(self.readings) < k:
            return 0.0

        last_k_readings = list(self.readings)[-k:]
        return sum(last_k_readings) / k

    def getMaxWindow(self, k: int) -> float:
        num_readings = len(self.readings)

        if k <= 0 or num_readings < k:
            return 0.0

        
        current_sum = sum(list(self.readings)[:k])
        max_avg = current_sum / k

        
        for i in range(k, num_readings):
            current_sum += self.readings[i] - self.readings[i - k]
            current_avg = current_sum / k
            
            if current_avg > max_avg:
                max_avg = current_avg

        return max_avg


if __name__ == "__main__":
    logger = TemperatureLogger()
    readings = [5, 6, 7, 8, 9, 10, 1, 2, 3, 4]

    print("Adding readings...")
    for temp in readings:
        logger.addReading(temp)
    
    print("Total readings in log:", len(logger.readings))
    print("---------------------------------------")
    
    k1 = 3
    print(f"Average of last {k1} readings: {logger.getAverage(k1):.2f}") 

    k2 = 5
    print(f"Average of last {k2} readings: {logger.getAverage(k2):.2f}") 
    print("---------------------------------------")
       
    k3 = 3
    print(f"Max average over any window of size {k3}: {logger.getMaxWindow(k3):.2f}")
    
    k4 = 5
    print(f"Max average over any window of size {k4}: {logger.getMaxWindow(k4):.2f}")
  
