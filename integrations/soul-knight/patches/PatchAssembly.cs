using System;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;

public class PatchAssembly {
    public static void Main(string[] args) {
        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(System.IO.Path.GetDirectoryName(args[0]));
        var asm = AssemblyDefinition.ReadAssembly(args[0], new ReaderParameters { AssemblyResolver = resolver });
        var bridge = AssemblyDefinition.ReadAssembly(args[1], new ReaderParameters { AssemblyResolver = resolver });
        var module = asm.MainModule;
        var bridgeType = bridge.MainModule.Types.First(t => t.Name == "HoningBridge");
        var inputType = bridge.MainModule.Types.First(t => t.Name == "HoningInput");
        var running = module.ImportReference(bridgeType.Fields.First(f => f.Name == "Running"));
        var bootstrap = module.ImportReference(bridgeType.Methods.First(m => m.Name == "Bootstrap"));
        int guarded = 0, inputs = 0;
        foreach (var type in module.Types) foreach (var method in type.Methods) {
            if (!method.HasBody) continue;
            var il = method.Body.GetILProcessor();
            if (method.Name == "Update" || method.Name == "FixedUpdate" || method.Name == "LateUpdate") {
                var first = method.Body.Instructions[0];
                il.InsertBefore(first, il.Create(OpCodes.Ldsfld, running));
                il.InsertBefore(first, il.Create(OpCodes.Brtrue, first));
                il.InsertBefore(first, il.Create(OpCodes.Ret));
                guarded++;
            }
            if (type.Name == "UIManager" && method.Name == "Awake") {
                foreach (var ret in method.Body.Instructions.Where(i => i.OpCode == OpCodes.Ret).ToArray())
                    il.InsertBefore(ret, il.Create(OpCodes.Call, bootstrap));
            }
            foreach (var instruction in method.Body.Instructions) {
                var called = instruction.Operand as MethodReference;
                if (called == null || called.DeclaringType.FullName != "UnityEngine.Input") continue;
                var replacement = inputType.Methods.FirstOrDefault(m => m.Name == called.Name);
                if (replacement == null) continue;
                instruction.Operand = module.ImportReference(replacement);
                inputs++;
            }
        }
        if (guarded != 15 || inputs != 12)
            throw new InvalidOperationException("Unexpected author assembly: refusing unverified patch targets");
        asm.Write(args[2]);
        Console.WriteLine("guarded_updates=" + guarded + " redirected_input_calls=" + inputs);
    }
}
